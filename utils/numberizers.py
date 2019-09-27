def numberize_labels(df, labels_column_name, base_label = None, numeric_labels = [1, -1], label_type=float):
	if base_label is None:
		labels = df[labels_column_name].unique()
		labels_dict = dict(zip(labels, map(lambda i: label_type(i), range(len(labels)))))
		return df.replace({labels_column_name: labels_dict})
	cdf = df.copy()
	cdf[labels_column_name] = df[labels_column_name].map({base_label: numeric_labels[0]}).fillna(numeric_labels[1]).astype(label_type)
	return cdf