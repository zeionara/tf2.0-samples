def numberize_labels(df, labels_column_name, base_label = None):
	if base_label is None:
		labels = df[labels_column_name].unique()
		labels_dict = dict(zip(labels, map(lambda i: float(i), range(len(labels)))))
		return df.replace({labels_column_name: labels_dict})
	cdf = df.copy()
	cdf[labels_column_name] = df[labels_column_name].map({base_label: 1}).fillna(-1).astype(float)
	return cdf