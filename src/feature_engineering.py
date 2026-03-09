def create_features(df):
    df['stress'] = df['load_applied'] / df['cross_section_area']
    df['slenderness_ratio'] = df['length'] / df['cross_section_area']
    df['load_factor'] = df['load_applied'] / df['yield_strength']
    df['safety_factor'] = df['yield_strength'] / df['stress']
    return df
