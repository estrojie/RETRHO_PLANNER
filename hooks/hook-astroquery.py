from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("astroquery", include_py_files=False)
