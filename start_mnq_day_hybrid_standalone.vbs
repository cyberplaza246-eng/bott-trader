Option Explicit
Dim sh, fso, root, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = root
cmd = "cmd /k title MNQ day hybrid & """ & root & "\start_mnq_day_hybrid.bat"""
sh.Run cmd, 1, False
