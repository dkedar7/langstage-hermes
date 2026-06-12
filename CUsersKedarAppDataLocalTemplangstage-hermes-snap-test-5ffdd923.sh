declare -x DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus"
declare -x DISPLAY=":0"
declare -x HOME="/home/dkedar"
declare -x HOSTTYPE="x86_64"
declare -x LANG="C.UTF-8"
declare -x LOGNAME="dkedar"
declare -x NAME="WIN-NCBE811APBL"
declare -x OLDPWD
declare -x PATH="/home/dkedar/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/lib/wsl/lib:/mnt/c/WINDOWS/system32:/mnt/c/WINDOWS:/mnt/c/WINDOWS/System32/Wbem:/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/:/mnt/c/WINDOWS/System32/OpenSSH/:/mnt/c/Program Files/Git/cmd:/mnt/c/Program Files/nodejs/:/mnt/c/Program Files/GitHub CLI/:/mnt/c/Users/Kedar/AppData/Local/hermes/hermes-agent/venv/Scripts:/mnt/c/Users/Kedar/AppData/Local/Programs/Python/Python311/Scripts/:/mnt/c/Users/Kedar/AppData/Local/Programs/Python/Python311/:/mnt/c/Users/Kedar/AppData/Local/Programs/Python/Launcher/:/mnt/c/Users/Kedar/AppData/Local/Programs/Quarto/bin:/mnt/c/Users/Kedar/AppData/Local/Microsoft/WindowsApps:/mnt/c/Users/Kedar/AppData/Local/Programs/Microsoft VS Code/bin:/mnt/c/users/kedar/.local/bin:/mnt/c/users/kedar/appdata/local/packages/pythonsoftwarefoundation.python.3.12_qbz5n2kfra8p0/localcache/local-packages/python312/scripts:/mnt/c/Users/Kedar/AppData/Local/Programs/Python/Python313:/mnt/c/Users/Kedar/AppData/Local/Programs/Python/Python312:/mnt/c/Program Files (x86)/chrome-win64:/mnt/c/Program Files (x86)/chromedriver-win64:/mnt/c/Users/Kedar/AppData/Roaming/npm:/mnt/c/Users/Kedar/AppData/Local/Microsoft/WinGet/Links:/mnt/c/Users/Kedar/bin:/mnt/c/Users/Kedar/.bun/bin:/mnt/c/Users/Kedar/.fly/bin:/snap/bin"
declare -x PULSE_SERVER="unix:/mnt/wslg/PulseServer"
declare -x PWD="/mnt/c/Users/Kedar/Documents/Code/deepagent-hermes"
declare -x SHELL="/bin/bash"
declare -x SHLVL="1"
declare -x TERM="xterm-256color"
declare -x USER="dkedar"
declare -x WAYLAND_DISPLAY="wayland-0"
declare -x WSL2_GUI_APPS_ENABLED="1"
declare -x WSLENV="WT_SESSION:WT_PROFILE_ID:"
declare -x WSL_DISTRO_NAME="Ubuntu"
declare -x WSL_INTEROP="/run/WSL/694_interop"
declare -x WT_PROFILE_ID="{61c54bbd-c2c6-5271-96e7-009a87ff44bf}"
declare -x WT_SESSION="bccc6184-c0c1-452d-8903-3229ede62530"
declare -x XDG_DATA_DIRS="/usr/local/share:/usr/share:/var/lib/snapd/desktop"
declare -x XDG_RUNTIME_DIR="/run/user/1000"
gawklibpath_append () 
{ 
    [ -z "$AWKLIBPATH" ] && AWKLIBPATH=`gawk 'BEGIN {print ENVIRON["AWKLIBPATH"]}'`;
    export AWKLIBPATH="$AWKLIBPATH:$*"
}
gawklibpath_default () 
{ 
    unset AWKLIBPATH;
    export AWKLIBPATH=`gawk 'BEGIN {print ENVIRON["AWKLIBPATH"]}'`
}
gawklibpath_prepend () 
{ 
    [ -z "$AWKLIBPATH" ] && AWKLIBPATH=`gawk 'BEGIN {print ENVIRON["AWKLIBPATH"]}'`;
    export AWKLIBPATH="$*:$AWKLIBPATH"
}
gawkpath_append () 
{ 
    [ -z "$AWKPATH" ] && AWKPATH=`gawk 'BEGIN {print ENVIRON["AWKPATH"]}'`;
    export AWKPATH="$AWKPATH:$*"
}
gawkpath_default () 
{ 
    unset AWKPATH;
    export AWKPATH=`gawk 'BEGIN {print ENVIRON["AWKPATH"]}'`
}
gawkpath_prepend () 
{ 
    [ -z "$AWKPATH" ] && AWKPATH=`gawk 'BEGIN {print ENVIRON["AWKPATH"]}'`;
    export AWKPATH="$*:$AWKPATH"
}
shopt -s expand_aliases
set +e
set +u
