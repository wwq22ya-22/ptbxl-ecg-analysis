"""Flash/debugger target selection must preserve the reference RAM and clock."""
import xml.etree.ElementTree as ET

import pytest

from scripts.package_serial_demo import configure_chip


@pytest.fixture
def target():
    return ET.fromstring('''<Project><Device>STM32F407VGTx</Device>
      <Cpu>IRAM(0x20000000,0x00020000) IRAM2(0x10000000,0x00010000) IROM(0x08000000,0x00100000) CLOCK(12000000)</Cpu>
      <FlashDriverDll>-FF0STM32F4xx_1024 -FL0100000 $$Device:STM32F407VGTx</FlashDriverDll>
      <OnChipMemories><IROM><Size>0x100000</Size></IROM><IRAM><Size>0x20000</Size></IRAM></OnChipMemories>
      <OCR_RVCT1><Size>0x100000</Size></OCR_RVCT1>
    </Project>''')


@pytest.mark.parametrize('chip,size,algorithm', [('STM32F407VE', 0x80000, '512'), ('STM32F407VG', 0x100000, '1024')])
def test_target_selection_and_roundtrip(target, chip, size, algorithm):
    configure_chip(target, chip)
    assert target.findtext('.//Device') == chip + 'Tx'
    assert int(target.findtext('.//IROM/Size'), 0) == size
    assert int(target.findtext('.//OCR_RVCT1/Size'), 0) == size
    assert f'IROM(0x08000000,0x{size:08X})' in target.findtext('.//Cpu')
    assert 'IRAM(0x20000000,0x00020000) IRAM2(0x10000000,0x00010000)' in target.findtext('.//Cpu')
    assert 'CLOCK(12000000)' in target.findtext('.//Cpu')
    assert target.findtext('.//IRAM/Size') == '0x20000'
    assert 'STM32F4xx_' + algorithm in target.findtext('.//FlashDriverDll')
    assert f'-FL0{size:06X}' in target.findtext('.//FlashDriverDll')
    configure_chip(target, 'STM32F407VG')
    assert '-FL0100000' in target.findtext('.//FlashDriverDll')
    assert 'STM32F4xx_1024' in target.findtext('.//FlashDriverDll')


def test_unknown_chip_is_rejected_before_configuration(target):
    original = ET.tostring(target)
    with pytest.raises(ValueError, match='Unsupported target'):
        configure_chip(target, 'STM32F401')
    assert ET.tostring(target) == original
