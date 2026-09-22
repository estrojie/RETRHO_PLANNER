#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

#define AppName "RETRHO Planner"
#define AppExeName "RHOPlanner.exe"
#define AppFolderName "RHOPlanner"

[Setup]
AppId={{5E5413C6-0774-4B68-9AC0-6BC84183835C}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=RETRHO Planner Project
DefaultDirName={localappdata}\Programs\{#AppFolderName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\..\release
OutputBaseFilename=RHOPlanner-Windows-x64-Setup
SetupIconFile=..\..\assets\rho_planner.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} installer

[Files]
Source: "..\..\dist\RHOPlanner\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
