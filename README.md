# Installation

Please follow the installation instructions to get started with the plugin.

## Dependencies
First of all, we need to install dependencies. Currently the only dependency is 
[librosa library](https://github.com/librosa/librosa "Librosa library github"). Blender has a bundled Python, so it 
needs to be installed inside it.

### Locate Blender's Python
It should be in *<blender-installation-directory>/<blender-version>/python/bin* directory. From now on, we will use 
*<blender-python>* to address this location.

### Installing *pip* for Blender's Python
* Download [pip](https://bootstrap.pypa.io/get-pip.py "Pip installation script") somewhere (e.g. to home directory)
* Run it with Blender's Python 
```shell
<blender-python>$ ./python3.5m ~/get-pip.py
```

### Installing *librosa* for Blender's Python
* Locate pip in blender. It should be in *<blender-installation-directory>/<blender-version>/python/lib/python3.5/site-packages*
directory. Lets call this directory *<blender-site-packages>*
* Then, from the <blender-python> directory run:
```shell
<blender-python>$ ./python3.5m <blender-site-packages>/pip install librosa 
```

## The Plugin
1. Clone / Download the source from [GitHub](https://github.com/Aspect26/MusicTo3D "Plugin's GitHub page")
2. Open Blender, and go to File -> User Preferences -> Add-ons -> Install Add-on from File... 
3. Choose the plugin's source (main.py)

# Usage

NOT FINISHED YET