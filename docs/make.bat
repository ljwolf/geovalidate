@ECHO OFF
pushd %~dp0
if "%SPHINXBUILD%" == "" (set SPHINXBUILD=sphinx-build)
set SOURCEDIR=source
set BUILDDIR=build
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.The 'sphinx-build' command was not found.
    exit /b 1
)
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end
:end
popd
