; Local Interpreter - Inno Setup script
;
; Builds the installer from dist\LocalInterpreter (see LocalInterpreter.spec).
; The application payload is self-contained: Python, PyQt6, CTranslate2, the
; CUDA 12 / cuDNN 9 runtime and the MSVC runtime all ship inside. The only
; thing fetched at install time is the Whisper weights (~4.6 GB), because
; embedding those would make the installer roughly five times larger.

#define AppName        "Local Interpreter"
#define AppVersion     "1.0.1"
#define AppPublisher   "Local Interpreter"
#define AppExe         "LocalInterpreter.exe"
#define FetchExe       "LocalInterpreterFetch.exe"
#define SourceDir      "dist\LocalInterpreter"

[Setup]
AppId={{7C3F1B62-4E1D-4D3A-9A0E-2C6B5E9A1F44}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
OutputDir=installer
; Deliberately not "setup.exe": Windows shims every executable with that name
; and lets it load version.dll & co. from its own folder, which is a classic
; DLL-hijacking vector for anything sitting in the Downloads directory.
OutputBaseFilename=LocalInterpreter-Setup-{#AppVersion}
SetupIconFile=assets\app.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
; ...but allow "just for me" so the app can also be installed without admin
PrivilegesRequiredOverridesAllowed=dialog commandline
DisableProgramGroupPage=yes
DisableDirPage=no
DiskSpanning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
; Only offered when the machine does not already have that model.
Name: "getmedium"; Description: "medium  —  1.5 GB, fastest, good accuracy"; GroupDescription: "Speech models (downloaded now, then used fully offline):"; Check: ModelMissing('medium')
Name: "getlarge"; Description: "large-v3  —  3.1 GB, best accuracy (recommended on an NVIDIA GPU)"; GroupDescription: "Speech models (downloaded now, then used fully offline):"; Check: ModelMissing('large-v3')

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{commonappdata}\LocalInterpreter\models"; Permissions: users-modify; Check: IsAdminInstallMode

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{#AppName} self-test"; Filename: "{app}\{#FetchExe}"; Parameters: "--self-test"; Comment: "Check the GPU, models and audio devices"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

; The models are several GB and are deliberately *not* listed here - removing
; them is offered as a question during uninstall instead (see [Code]).

[Code]
var
  ModelsDir: String;

// Reinstalling, upgrading, or installing next to an existing copy must not
// re-download several GB of weights. The Hugging Face cache is checked too,
// but only by the downloader itself - see local_model_dir() in the app.
function ModelMissing(Size: String): Boolean;
begin
  Result := not FileExists(ExpandConstant('{commonappdata}\LocalInterpreter\models\faster-whisper-'
                                          + Size + '\model.bin'))
        and not FileExists(ExpandConstant('{localappdata}\LocalInterpreter\models\faster-whisper-'
                                          + Size + '\model.bin'));
end;

function FetchModel(Size: String): Boolean;
var
  Code: Integer;
begin
  WizardForm.StatusLabel.Caption := 'Downloading the ' + Size + ' speech model...';
  Result := Exec(ExpandConstant('{app}\{#FetchExe}'),
                 '--fetch-models ' + Size + ' --dest="' + ModelsDir + '"',
                 ExpandConstant('{app}'), SW_SHOW, ewWaitUntilTerminated, Code)
            and (Code = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Failed: String;
begin
  if CurStep <> ssPostInstall then
    Exit;

  if IsAdminInstallMode() then
    ModelsDir := ExpandConstant('{commonappdata}\LocalInterpreter\models')
  else
    ModelsDir := ExpandConstant('{localappdata}\LocalInterpreter\models');
  if not DirExists(ModelsDir) then
    ForceDirectories(ModelsDir);

  Failed := '';
  if WizardIsTaskSelected('getmedium') then
    if not FetchModel('medium') then
      Failed := Failed + '  medium' + #13#10;
  if WizardIsTaskSelected('getlarge') then
    if not FetchModel('large-v3') then
      Failed := Failed + '  large-v3' + #13#10;

  if (Failed <> '') and not WizardSilent then
    MsgBox('These speech models could not be downloaded:' + #13#10 + #13#10 + Failed + #13#10 +
           'This is almost always a missing or blocked internet connection.' + #13#10 +
           'Local Interpreter is installed and will download the model it needs' + #13#10 +
           'the first time you press Start, so you can simply try again later.',
           mbInformation, MB_OK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Dir: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;

  if IsAdminInstallMode() then
    Dir := ExpandConstant('{commonappdata}\LocalInterpreter')
  else
    Dir := ExpandConstant('{localappdata}\LocalInterpreter');

  if not DirExists(Dir + '\models') then
    Exit;

  // Several GB that took a while to fetch - never remove them silently.
  if UninstallSilent or (MsgBox('Also delete the downloaded speech models?'
       + #13#10 + #13#10 + Dir + '\models' + #13#10 + #13#10
       + 'They can be several GB. Keep them if you plan to reinstall or if you '
       + 'run Local Interpreter from source.', mbConfirmation, MB_YESNO) = IDYES) then
    DelTree(Dir + '\models', True, True, True);
end;
