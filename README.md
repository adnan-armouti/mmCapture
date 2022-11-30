# mmcapture

This repository captures software-triggered raw multimodal (image, rf, audio) synchronous data from a sensor stack.

All code currently under development; please see disclaimer at the end of this README. The "main" branch has a more extensive set of sensors that record videos up to 30 seconds long (subject to RAM limitations), while the "low_light" branch is optimized for a smaller set of sensors that record data per frame (bypassing RAM limitations).

Please note: Due to Python's Global Interpretor Lock (GIL) a true multithreaded approach is not possible. Please refer to this [repository](https://github.com/UCLA-VMG/syndicate) for a multithreaded C++ implementation. The use of a hardware trigger system via GPIO is also recommended for microsecond-level synchronization across sensors per frame; more information on this can be found at this [link](https://www.flir.com/support-center/iis/machine-vision/application-note/configuring-synchronized-capture-with-multiple-cameras/). 

## Python Installation

**(1) Recommended Installation**

Install Python. Currently we support Python 3.6 only. To download Python, visit https://www.python.org/downloads/. Note that the Python website defaults to 32-bit interpreters, so if you want a 64-bit version of Python you have to click into the specific release version. For ease of installation, we recommend installing python through anaconda distribution, via https://www.anaconda.com/products/distribution

**(2) PATH**

(Optional) Set the PATH environment variable for your Python installation. This may have been done automatically as part of installation, but to do this manually you have to open Environment Variables through the following:

My Computer > Properties > Advanced System Settings > Environment Variables

Add your Python installation location to your PATH variable. For example, if you installed Python at C:\Python36\, you would add the following entry to the PATH variable:

C:\Python36\<rest_of_path>

Note that the anaconda distribution does this for you automatically.

<hr /> 

## Hardware 

**(1) MX800**

Please refer to the following [link](https://compatibility.rockwellautomation.com/Pages/MultiProductFindDownloads.aspx?crumb=112&mode=3&refSoft=1&versions=59657). The link will redirect to the software we have utilized to connect to the MX800 through the Ethernet port. A configuration named mx800.bpc has been provided in data_acquisition _/sensors/configs_ folder. However, you may need to regenerate one for your specific system.

Once connected, please clone this [GitHub Repository](https://github.com/xeonfusion/VSCaptureMP) that contains the C# code to collect data from the MX800. You will need to compile the C# code with a tool such as [Visual Studio](https://visualstudio.microsoft.com/). The generated binaries will need to be linked with the mx800_sensor.py file. The binaries can be linked through the \_\__init_\_\_ function of the _MX800\_Sensor_ class.

**(2) RGBD Camera**

This code was developed using the Zed Camera; please follow these [instructions](https://www.stereolabs.com/docs/get-started-with-zed/) provided by StereoLabs.

The  _sensors/rgbd\_sensor.py_ file provided is for the Zed Camera used in the paper. For any other camera, please use _sensors/rgb\_sensor.py_.

**(3) Other sensors**

Other sensors used: FLIR Boson Radiometric camera for thermal data, FLIR Grasshopper3 for NIR data, and modified Canon EOS 60D for uv data, FLIR Blackfly S for grayscale polarized data.

<hr/>

## mmcapture Usage Guide

Please run the sync_sensors.py file.
<hr /> 

## Disclaimer

This codebase is still in development, and highly specific to our multimodal sensor stack. Use of this code on other hardware may produce bugs.

<hr /> 




