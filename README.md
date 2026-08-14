# Raspberry Pi Spectrometer DAQ

A data acquisition and GUI for data from a Raspberry Pi Spectrometer (https://github.com/leswright1977/PySpectrometer2). Built whilst I was at CERN for a summer project.

## Startup
The login logic is weird due to quirks with the raspberry pi zero login logic. After investigation data has to be sent on a separate serial connection. During startup, the pi is logged in and pi_sender.py is launched. The Pi Camera gain is then sent which then triggers the start of data being sent. If data is already being sent from the Pi (due to it already being logged in and running), the startup is skipped.

## GUI
The GUI plots data, has functionality to find peaks and smooths the data. 

## Calibration
To calibrate, press the calibration button, and click on a known peak. Then enter the true wavelength in the popup which saves a calibration point. On startup a fit is performed to get a relationship between pixel number and wavelength. Ideally 3+ calibration points should be done.



