@echo off
REM Create deployment package for Lambda
echo Creating Lambda deployment package...

REM Create temp directory
if exist temp_deploy rmdir /s /q temp_deploy
mkdir temp_deploy

REM Copy Python file
copy app.py temp_deploy\

REM Create zip file (requires PowerShell)
cd temp_deploy
powershell -command "Compress-Archive -Path * -DestinationPath ..\lambda-deployment.zip -Force"
cd ..

REM Clean up
rmdir /s /q temp_deploy

echo Deployment package created: lambda-deployment.zip
echo.
echo To deploy, run:
echo aws lambda update-function-code --function-name YOUR_FUNCTION_NAME --zip-file fileb://lambda-deployment.zip
echo.
set /p FUNCTION_NAME="Enter your Lambda function name: "
if not "%FUNCTION_NAME%"=="" (
    echo Deploying to function: %FUNCTION_NAME%
    aws lambda update-function-code --function-name %FUNCTION_NAME% --zip-file fileb://lambda-deployment.zip
    if %ERRORLEVEL% EQU 0 (
        echo Deployment successful!
    ) else (
        echo Deployment failed!
    )
)
pause