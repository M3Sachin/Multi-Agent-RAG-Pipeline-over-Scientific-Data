import os
import ast
import subprocess
import tempfile

# Forbidden modules and functions for security
FORBIDDEN_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "telnet",
    "shutil",
    "glob",
    "importlib",
    "__import__",
    "pickle",
    "marshal",
    "exec",
    "eval",
    "compile",
    "open",
    "file",
    "input",
}

FORBIDDEN_FUNCTIONS = {
    "eval",
    "exec",
    "compile",
    "open",
    "exit",
    "quit",
    "breakpoint",
    "help",
    "dir",
    "vars",
    "locals",
    "globals",
}

TIMEOUT_SECONDS = 10


def validate_code(code: str) -> tuple[bool, str]:
    """
    Validate Python code for safety using AST analysis.

    Returns:
        (is_safe, error_message)
    """
    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name in FORBIDDEN_MODULES:
                        return False, f"Import of '{module_name}' is forbidden"

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name in FORBIDDEN_MODULES:
                        return False, f"Import from '{module_name}' is forbidden"

            # Check function calls
            elif isinstance(node, ast.Call):
                # Check for forbidden builtins
                if isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_FUNCTIONS:
                        return False, f"Function '{node.func.id}' is forbidden"
                # Check for attribute calls
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in FORBIDDEN_FUNCTIONS:
                        return False, f"Function '{node.func.attr}' is forbidden"

            # Check for subprocess/commands
            elif isinstance(node, ast.Attribute):
                if node.attr in ["system", "popen", "spawn"]:
                    return False, f"Attribute '{node.attr}' is forbidden"

        return True, ""

    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def execute_python(code: str) -> dict:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute

    Returns:
        dict with success, output, error
    """
    # Validate code first
    is_safe, error_msg = validate_code(code)
    if not is_safe:
        return {"success": False, "output": "", "error": error_msg, "execution_time": 0}

    # Create temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Execute with timeout
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"

        if not output:
            output = "Code executed successfully with no output."

        return {
            "success": result.returncode == 0,
            "output": output,
            "error": result.stderr if result.returncode != 0 else "",
            "execution_time": TIMEOUT_SECONDS,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Execution timed out after {TIMEOUT_SECONDS} seconds",
            "execution_time": TIMEOUT_SECONDS,
        }

    except Exception as e:
        return {"success": False, "output": "", "error": str(e), "execution_time": 0}

    finally:
        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass
