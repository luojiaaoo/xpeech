# DT and ACPI Validation

## Device Tree
```bash
dtc -I fs -O dts /sys/firmware/devicetree/base > running.dts
make dt_binding_check DT_SCHEMA_FILES=<binding>.yaml
make dtbs_check DT_SCHEMA_FILES=<binding>.yaml
```

Check: `compatible`, `reg`, `interrupts`, `clocks`, `resets`, supplies, `pinctrl`, `power-domains`, DMA properties, `status`.

## ACPI
```bash
sudo acpidump -o acpi.dat
iasl -d acpi.dat
grep -R "HID\|CID\|CRS\|DSD\|STA" -n *.dsl
```

Check: `_HID`/`_CID`, `_CRS`, `_STA`, `_DSD`, GPIO/I2C/SPI descriptors, power resources.
