# Subsystem Command Recipes

## PCIe
```bash
lspci -nnvv
lspci -t
dmesg -T | grep -Ei 'pci|pcie|aer|aspm|link'
```

## USB
```bash
lsusb -t
lsusb -v
dmesg -T | grep -Ei 'usb|xhci|dwc|phy|typec'
```

## I2C
```bash
i2cdetect -l
sudo i2ctransfer -y <bus> w1@<addr> <reg> r1
```

## SPI
```bash
ls /sys/bus/spi/devices
dmesg -T | grep -Ei 'spi|mtd|nor|nand'
```

## Regulators/Clocks/Pinctrl
```bash
sudo cat /sys/kernel/debug/regulator/regulator_summary
sudo cat /sys/kernel/debug/clk/clk_summary
sudo cat /sys/kernel/debug/pinctrl/*/pinmux-pins
```

## Dynamic Debug
```bash
sudo sh -c 'echo "file drivers/path/file.c +p" > /sys/kernel/debug/dynamic_debug/control'
```
