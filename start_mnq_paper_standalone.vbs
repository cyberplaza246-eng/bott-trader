' Close Lucid desktop. Do not start a second copy. Close Cursor is OK — leave this black window open.
Option Explicit
Dim sh, fso, root, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = root
' WindowStyle 1 = normal visible console. WaitOnReturn False = detached (survives Cursor).
cmd = "cmd /k title MNQ overnight paper & """ & root & "\start_mnq_overnight_paper.bat"""
sh.Run cmd, 1, False
