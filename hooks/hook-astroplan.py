from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = []
datas += collect_data_files("astroplan", include_py_files=False)

# Ensure importlib.metadata can find these in a frozen app
datas += copy_metadata("matplotlib")
datas += copy_metadata("astropy")
datas += copy_metadata("astroplan")
