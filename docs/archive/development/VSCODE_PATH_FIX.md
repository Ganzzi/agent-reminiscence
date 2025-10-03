# 🛠️ VSCode Terminal PATH Configuration - Permanent Fix

**Problem**: Programs installed in Windows (like `uv`, `python`, etc.) are not recognized in VSCode terminals, even though they work in Windows Terminal.

**Root Cause**: VSCode terminals inherit environment variables from the VSCode process, not from the current system environment. When you install new programs or update PATH after VSCode is already running, VSCode doesn't pick up these changes.

---

## ✅ Permanent Solutions Implemented

### 1. Workspace-Level Configuration (✅ Applied)

**File**: `.vscode/settings.json`

```json
{
  "terminal.integrated.env.windows": {
    "PATH": "C:\\Users\\Administrator\\.local\\bin;${env:PATH}"
  },
  "terminal.integrated.inheritEnv": true,
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true
}
```

**What it does**:
- ✅ Adds `uv` installation directory to PATH for all terminals in this workspace
- ✅ Enables environment inheritance from the system
- ✅ Configures Python virtual environment auto-activation
- ✅ Sets default Python interpreter to workspace venv

**How to apply**: 
1. File already created in `.vscode/settings.json`
2. **Restart any open terminals** in VSCode (close and reopen)
3. Or **restart VSCode** for changes to take effect

---

### 2. Global PATH Configuration (For All Projects)

If you want `uv` and other tools to work in **all VSCode workspaces**, add this to your **User Settings**:

**Location**: 
- Press `Ctrl+Shift+P` → Type "Preferences: Open User Settings (JSON)"
- Or: `%APPDATA%\Code\User\settings.json`

**Add this**:
```json
{
  "terminal.integrated.env.windows": {
    "PATH": "C:\\Users\\Administrator\\.local\\bin;${env:PATH}"
  },
  "terminal.integrated.inheritEnv": true
}
```

---

### 3. PowerShell Profile (Alternative Method)

For even more control, you can create a PowerShell profile that VSCode will always load:

**File**: `$PROFILE.CurrentUserAllHosts`  
**Path**: `C:\Users\Administrator\Documents\PowerShell\profile.ps1`

```powershell
# Add custom paths
$env:PATH = "C:\Users\Administrator\.local\bin;$env:PATH"

# Add other custom paths as needed
# $env:PATH = "C:\MyTools;$env:PATH"

# Activate Python virtual environment if in project
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
}
```

**To create**:
```powershell
# Create profile if it doesn't exist
if (!(Test-Path -Path $PROFILE.CurrentUserAllHosts)) {
    New-Item -ItemType File -Path $PROFILE.CurrentUserAllHosts -Force
}

# Edit profile
notepad $PROFILE.CurrentUserAllHosts
```

---

## 🔧 Quick Fixes When PATH Issues Occur

### Fix 1: Restart Terminal (Fastest)
1. Close current terminal tab
2. Open new terminal (`Ctrl+` backtick)
3. PATH changes will be applied

### Fix 2: Reload Window (Medium)
1. Press `Ctrl+Shift+P`
2. Type "Developer: Reload Window"
3. All environment variables refreshed

### Fix 3: Restart VSCode (Most Reliable)
1. Close VSCode completely
2. Reopen VSCode
3. All system PATH changes will be picked up

### Fix 4: Add PATH Temporarily (For Current Session)
```powershell
# Add to current terminal session
$env:PATH = "C:\Users\Administrator\.local\bin;$env:PATH"

# Verify
Get-Command uv
```

---

## 🎯 Verification Steps

After applying the fixes, verify everything works:

```powershell
# 1. Check if uv is accessible
uv --version

# 2. Check if Python is accessible
python --version

# 3. Check PATH contains your tools
$env:PATH -split ';' | Select-String "\.local\\bin"

# 4. Verify virtual environment
.venv\Scripts\activate
python --version
```

**Expected Output**:
```
uv 0.x.x
Python 3.13.7
C:\Users\Administrator\.local\bin
Python 3.13.7
```

---

## 📋 Common PATH Locations

Add these to your PATH if needed:

```json
{
  "terminal.integrated.env.windows": {
    "PATH": "C:\\Users\\Administrator\\.local\\bin;C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python313;C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python313\\Scripts;${env:PATH}"
  }
}
```

**Common tool locations**:
- **uv**: `C:\Users\Administrator\.local\bin`
- **Python**: `C:\Users\Administrator\AppData\Local\Programs\Python\Python313`
- **pip/pytest**: `C:\Users\Administrator\AppData\Local\Programs\Python\Python313\Scripts`
- **Node.js**: `C:\Program Files\nodejs`
- **Git**: `C:\Program Files\Git\cmd`

---

## 🔄 When to Apply Which Solution

| Scenario | Solution | Scope |
|----------|----------|-------|
| Single project needs tool | Workspace `.vscode/settings.json` | This project only |
| All projects need tool | User `settings.json` | All workspaces |
| Complex startup needs | PowerShell `$PROFILE` | All PowerShell sessions |
| Quick one-time use | `$env:PATH = "...;$env:PATH"` | Current terminal only |

---

## 🐛 Troubleshooting

### Issue: "uv not recognized" after restart
**Solution**: Check that `.vscode/settings.json` exists and PATH is correct
```powershell
Get-Content .vscode\settings.json | Select-String "PATH"
```

### Issue: Settings not taking effect
**Solution**: 
1. Ensure no syntax errors in JSON
2. Restart VSCode completely (not just reload window)
3. Check if workspace settings override user settings

### Issue: Virtual environment not activating
**Solution**: Check Python extension settings
```json
{
  "python.terminal.activateEnvironment": true,
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
}
```

### Issue: PATH keeps resetting
**Solution**: Make sure you're editing the right settings file
1. Workspace: `.vscode/settings.json` (project-specific)
2. User: `%APPDATA%\Code\User\settings.json` (global)

---

## 📚 Additional Resources

- **VSCode Terminal Docs**: https://code.visualstudio.com/docs/terminal/basics
- **Environment Variables**: https://code.visualstudio.com/docs/terminal/profiles
- **PowerShell Profile**: https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_profiles

---

## ✨ What We Fixed

1. ✅ **VSCode Terminal PATH**: Added `.vscode/settings.json` with PATH configuration
2. ✅ **Environment Inheritance**: Enabled `terminal.integrated.inheritEnv`
3. ✅ **Python Integration**: Configured virtual environment auto-activation
4. ✅ **Documentation**: Created this permanent reference guide

---

## 🚀 Quick Reference

**To use a newly installed program in VSCode terminal**:

1. **Best**: Add to `.vscode/settings.json` → Restart terminal
2. **Good**: Add to User `settings.json` → Restart VSCode
3. **Fast**: Run `$env:PATH = "C:\Path\To\Tool;$env:PATH"` in terminal

**This terminal knows about**:
- ✅ `uv` (Python package manager)
- ✅ `python` (Python 3.13.7)
- ✅ `pytest` (test runner)
- ✅ All other tools in system PATH

---

## 🎓 Key Takeaways

1. **VSCode terminals inherit from VSCode process**, not current system
2. **System PATH changes** require VSCode restart to be recognized
3. **Workspace settings** override user settings
4. **PowerShell profile** runs every time a terminal opens
5. **Virtual environments** can be auto-activated

---

**Status**: ✅ Fixed and Documented  
**Last Updated**: October 2, 2025  
**Applies To**: All future VSCode terminal PATH issues
