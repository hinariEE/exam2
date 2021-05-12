# exam2
## Setup environment
### For the accelerometer
```
mbed add http://developer.mbed.org/teams/ST/code/BSP_B-L475E-IOT01/#bfe8272ced90
rm BSP_B-L475E-IOT01/Drivers/BSP/B-L475E-IOT01/stm32l475e_iot01_qspi.*
```
### For RPC
```
mbed add https://gitlab.larc-nthu.net/ee2405_2021/mbed_rpc.git
```
### For WiFi
```
cd ~/ee2405/mbed-os
mbed add wifi-ism43362
```
create mbed_app.json
### For MQTT
```
mbed add https://gitlab.larc-nthu.net/ee2405_2019/wifi_mqtt.git
rm wifi_mqtt/main.cpp
```
revise wifi_mqtt/MQTTNetwork.h according to ee2405
### compile command
```
sudo mbed compile --source . --source ~/ee2405/mbed-os/ -m B_L4S5I_IOT01A -t GCC_ARM -f
```
