# Chinese to English column mappings for reservoir data
RESERVOIR_COLUMNS_MAP = {
    '日期': 'date',
    '库(淀)': 'reservoir',
    '库站': 'station',
    '正常蓄水位(m)': 'normal_water_level_m',
    '水位08h(m)': 'water_level_08h_m',
    '蓄水量08h(百万m³)': 'storage_08h_million_m3',
    '出库流量08h(m³/s)': 'outflow_08h_m3s',
    '闸门启闭-孔数': 'gate_holes',
    '闸门启闭-开启高度': 'gate_opening_height',
    '日均入库流量08h(m³/s)': 'avg_daily_inflow_08h_m3s',
    '日均出库流量08h(m³/s)': 'avg_daily_outflow_08h_m3s'
}

# List of numeric columns for type conversion
NUMERIC_COLUMNS = [
    'normal_water_level_m',
    'water_level_08h_m',
    'storage_08h_million_m3',
    'outflow_08h_m3s',
    'gate_holes',
    'gate_opening_height',
    'avg_daily_inflow_08h_m3s',
    'avg_daily_outflow_08h_m3s'
] 