python -c "import rasterio; src=rasterio.open('/home/wyr/code/martin/martin/data/s1/S1_2019_Q1.tif'); print('波段数:', src.count); print('各波段名称/描述:', src.descriptions)"
