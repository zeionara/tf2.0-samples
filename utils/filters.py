def filter_by_label(df, labels_column_name, base_label):
	return df[df[labels_column_name] == base_label]

def split_by_label_values(df, labels_column_name):
	return {label: filter_by_label(df, labels_column_name, label) for label in df[labels_column_name].unique()}