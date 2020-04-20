# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['reconocimiento.py'],
             pathex=['/home/laumecha/Documents/TrabajoSMI/reconocimientofacialOpencv'],
             binaries=[],
             datas=[('/usr/lib/python3/dist-packages/PIL/','PIL'),],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
a.datas += Tree('/home/laumecha/.virtualenvs/facecourse-py3/lib/python3.7/site-packages/sklearn/', prefix='sklearn')
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='reconocimiento',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
