// Jamie's Reminders helper: `reminders-helper status|list|create|complete`.
// Reads one JSON object on stdin, writes one JSON object on stdout, and always exits 0
// for handled errors so the caller can read the structured error.
import EventKit
import Foundation

struct Failure: Error {
    let code: String
    let message: String
}

let store = EKEventStore()

func reply(_ object: [String: Any]) -> Never {
    let data = (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{}".utf8)
    FileHandle.standardOutput.write(data)
    exit(0)
}

func accessName() -> String {
    let status = EKEventStore.authorizationStatus(for: .reminder)
    if #available(macOS 14.0, *) {
        switch status {
        case .fullAccess: return "granted"
        case .writeOnly: return "write_only"
        case .denied: return "denied"
        case .restricted: return "restricted"
        default: return "not_determined"
        }
    }
    switch status {
    case .authorized: return "granted"
    case .denied: return "denied"
    case .restricted: return "restricted"
    default: return "not_determined"
    }
}

func requireAccess() throws {
    switch accessName() {
    case "granted": return
    case "denied", "restricted", "write_only":
        throw Failure(
            code: "permission_denied",
            message: "Reminders access is denied. Allow it in System Settings > Privacy & Security > Reminders."
        )
    default: break
    }
    let semaphore = DispatchSemaphore(value: 0)
    var granted = false
    if #available(macOS 14.0, *) {
        store.requestFullAccessToReminders { ok, _ in granted = ok; semaphore.signal() }
    } else {
        store.requestAccess(to: .reminder) { ok, _ in granted = ok; semaphore.signal() }
    }
    semaphore.wait()
    if !granted {
        throw Failure(code: "permission_denied", message: "Reminders access was not granted.")
    }
}

let isoFormatter: ISO8601DateFormatter = {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    return formatter
}()

func encode(_ reminder: EKReminder) -> [String: Any] {
    var item: [String: Any] = [
        "id": reminder.calendarItemIdentifier,
        "title": reminder.title ?? "",
        "list": reminder.calendar?.title ?? "",
        "completed": reminder.isCompleted,
    ]
    if let components = reminder.dueDateComponents {
        if components.hour == nil, let year = components.year, let month = components.month,
            let day = components.day
        {
            item["due"] = String(format: "%04d-%02d-%02d", year, month, day)
        } else if let date = Calendar.current.date(from: components) {
            item["due"] = isoFormatter.string(from: date)
        }
    }
    if let notes = reminder.notes, !notes.isEmpty { item["notes"] = notes }
    return item
}

func calendars(named name: String?) throws -> [EKCalendar]? {
    guard let name = name, !name.isEmpty else { return nil }
    let matches = store.calendars(for: .reminder).filter { $0.title == name }
    if matches.isEmpty { throw Failure(code: "not_found", message: "No Reminders list named \(name).") }
    return matches
}

func list(_ request: [String: Any]) throws -> [String: Any] {
    try requireAccess()
    let includeCompleted = request["include_completed"] as? Bool ?? false
    let limit = request["limit"] as? Int ?? 25
    let lists = try calendars(named: request["list"] as? String)
    let predicate =
        includeCompleted
        ? store.predicateForReminders(in: lists)
        : store.predicateForIncompleteReminders(
            withDueDateStarting: nil, ending: nil, calendars: lists)
    let semaphore = DispatchSemaphore(value: 0)
    var found: [EKReminder] = []
    store.fetchReminders(matching: predicate) { reminders in
        found = reminders ?? []
        semaphore.signal()
    }
    semaphore.wait()
    let sorted = found.sorted {
        let a = $0.dueDateComponents.flatMap { Calendar.current.date(from: $0) } ?? .distantFuture
        let b = $1.dueDateComponents.flatMap { Calendar.current.date(from: $0) } ?? .distantFuture
        return a < b
    }
    return ["items": sorted.prefix(max(1, limit)).map(encode)]
}

func create(_ request: [String: Any]) throws -> [String: Any] {
    try requireAccess()
    guard let title = request["title"] as? String, !title.isEmpty else {
        throw Failure(code: "invalid_request", message: "A reminder title is required.")
    }
    let reminder = EKReminder(eventStore: store)
    reminder.title = title
    reminder.notes = request["notes"] as? String
    if let lists = try calendars(named: request["list"] as? String) {
        reminder.calendar = lists[0]
    } else if let fallback = store.defaultCalendarForNewReminders() {
        reminder.calendar = fallback
    } else {
        throw Failure(code: "not_found", message: "There is no default Reminders list.")
    }
    if let due = request["due"] as? String {
        if request["all_day"] as? Bool ?? false {
            let parts = due.split(separator: "-").compactMap { Int($0) }
            guard parts.count == 3 else {
                throw Failure(code: "invalid_request", message: "Invalid due date \(due).")
            }
            reminder.dueDateComponents = DateComponents(
                year: parts[0], month: parts[1], day: parts[2])
        } else {
            guard let date = isoFormatter.date(from: due) else {
                throw Failure(code: "invalid_request", message: "Invalid due time \(due).")
            }
            reminder.dueDateComponents = Calendar.current.dateComponents(
                [.year, .month, .day, .hour, .minute, .timeZone], from: date)
            reminder.addAlarm(EKAlarm(absoluteDate: date))
        }
    }
    try store.save(reminder, commit: true)
    return encode(reminder)
}

func complete(_ request: [String: Any]) throws -> [String: Any] {
    try requireAccess()
    guard let id = request["id"] as? String,
        let reminder = store.calendarItem(withIdentifier: id) as? EKReminder
    else {
        throw Failure(code: "not_found", message: "Reminder not found.")
    }
    reminder.isCompleted = true
    try store.save(reminder, commit: true)
    return encode(reminder)
}

let arguments = CommandLine.arguments
guard arguments.count == 2 else {
    reply(["ok": false, "error": ["code": "usage", "message": "usage: reminders-helper status|list|create|complete"]])
}
let input = FileHandle.standardInput.readDataToEndOfFile()
let request = (try? JSONSerialization.jsonObject(with: input)) as? [String: Any] ?? [:]

do {
    let result: [String: Any]
    switch arguments[1] {
    case "status": result = ["access": accessName()]
    case "list": result = try list(request)
    case "create": result = try create(request)
    case "complete": result = try complete(request)
    default: throw Failure(code: "usage", message: "Unknown command \(arguments[1]).")
    }
    reply(["ok": true, "result": result])
} catch let failure as Failure {
    reply(["ok": false, "error": ["code": failure.code, "message": failure.message]])
} catch {
    reply(["ok": false, "error": ["code": "error", "message": error.localizedDescription]])
}
