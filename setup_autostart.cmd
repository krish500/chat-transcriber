@echo off
rem registers the chat transcriber to start in the background every time you log in
schtasks /create /f /tn "Chat Transcriber" /sc onlogon /rl limited /tr "pythonw \"%USERPROFILE%\Videos\chat-transcriber\transcribe_chats.py\" \"%USERPROFILE%\Videos\" --watch --log \"%USERPROFILE%\Videos\chat-transcriber\transcriber.log\""
rem start it now too so you don't have to log out and back in
schtasks /run /tn "Chat Transcriber"
echo.
echo Done. New recordings in your Videos folder will be transcribed automatically.
echo Log: %USERPROFILE%\Videos\chat-transcriber\transcriber.log
pause
