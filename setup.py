from cx_Freeze import setup, Executable

setup(
    name="UGAD+ AI",
    version="1.0",
    description="Aplicativo UGAD+ com IA",
    executables=[Executable("run.py")]
)
