#define MyAppName "三角洲下机助手"
#define MyAppVersion GetEnv("DELTA_EXIT_ASSISTANT_VERSION")
#if MyAppVersion == ""
  #define MyAppVersion "2.3.5"
#endif
#define MyAppPublisher "Delta Exit Assistant"
#define MyAppExeName "Delta-Exit-Assistant.exe"
#define AppSourceDir "..\\release\\app\\Delta-Exit-Assistant"

[Setup]
AppId={{8E66B50A-1871-47B9-844A-6D7821CF8BE3}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Delta Exit Assistant
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\release
OutputBaseFilename=Delta-Exit-Assistant-Setup-{#MyAppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "其他选项："; Flags: unchecked

[Files]
Source: "{#AppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\release\Uninstall-Delta-Exit-Assistant.exe"; DestDir: "{app}"; DestName: "一键卸载三角洲下机助手.exe"; Flags: ignoreversion

[Icons]
Name: "{app}\启动三角洲下机助手"; Filename: "{app}\{#MyAppExeName}"
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
