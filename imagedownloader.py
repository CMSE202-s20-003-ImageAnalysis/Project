import wget
		
devices = ['A01_GioneeS55', 'A02_Huawei-P8', 'A03_Huawei-P9', 'A04_Huawei-P10', 'A05_Hwawie-MatePro10', 'A06_Huawei-Y5', 'A07_Galaxy-S7', 'A08_Galaxy-S7', 'A09_Galaxy-Note5', 'A10_Galaxy-J7', 'A11_Xiaomi5', 'A12_Huawei-RY6', 'A13_Huawei-RY6', 'A14_Xiaomi-5A', 'A15_Xiaom-3', 'A16_OnePlus-3t', 'A17_AsusZenfone-2', 'I03_iPhone7', 'I04_iPad-Air', 'I06_iPhone-5S']

category = "HDR"
for device in devices[-1:]:
    number = 1
    while True:
        try:
            link = 'ftp://lesc.dinfo.unifi.it/pub/Public/HDR/{}/NAT/{}_TRIPOD/{}_{}_TRIPOD_{:03}.jpg'.format(device, category, device[:3], category, number)
            wget.download(link, out=f"{device}/{category}-{number}.jpg")
            number += 1
        except Exception as thing:
            print(f"Exception Occurred at : {category}-{number}")
            if category == "HDR":
                category = "SDR"
                number = 1
            else:
                break
    
     