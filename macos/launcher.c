/*
 * Jamie.app's executable. It runs `python -m assistant gui` inside this process,
 * so macOS shows Jamie (not Python) in the Dock and names Jamie in permission prompts.
 * JAMIE_PYTHON is set when the app is built; see build-app.sh.
 */
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void log_to_file(const char *home) {
    char path[1024];
    if (home == NULL || isatty(STDERR_FILENO)) {
        return;  /* started from a terminal: keep its output there */
    }
    snprintf(path, sizeof path, "%s/Library/Logs/Jamie", home);
    mkdir(path, 0755);
    snprintf(path, sizeof path, "%s/Library/Logs/Jamie/app.log", home);
    if (freopen(path, "a", stderr) != NULL) {
        dup2(fileno(stderr), STDOUT_FILENO);
    }
}

int main(int argc, char **argv) {
    PyStatus status;
    PyConfig config;
    /* `jamie gui` plus the app's own arguments, such as --hidden from the login item. */
    char **arguments = calloc((size_t)argc + 2, sizeof *arguments);
    int count = 0;
    const char *home = getenv("HOME");

    if (arguments == NULL) {
        return 1;
    }
    arguments[count++] = argv[0];
    arguments[count++] = "gui";
    for (int index = 1; index < argc; index++) {
        if (strncmp(argv[index], "-psn_", 5) != 0) {  /* older macOS adds a process serial */
            arguments[count++] = argv[index];
        }
    }
    log_to_file(home);
    /* Finder starts apps in /; work from the home folder, as a new terminal does. */
    if (home != NULL && chdir(home) != 0) {
        perror("chdir");
    }

    PyConfig_InitPythonConfig(&config);
    config.parse_argv = 0;
    /* The virtual environment's interpreter, so its pyvenv.cfg selects its packages. */
    status = PyConfig_SetBytesString(&config, &config.executable, JAMIE_PYTHON);
    if (PyStatus_Exception(status)) goto fail;
    status = PyConfig_SetString(&config, &config.run_module, L"assistant");
    if (PyStatus_Exception(status)) goto fail;
    status = PyConfig_SetBytesArgv(&config, count, arguments);
    if (PyStatus_Exception(status)) goto fail;
    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) goto fail;
    PyConfig_Clear(&config);
    return Py_RunMain();

fail:
    PyConfig_Clear(&config);
    Py_ExitStatusException(status);
}
