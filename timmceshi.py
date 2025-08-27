import timm
feature_extractor = timm.create_model('lcnet_075',features_only='Ture')

print('type:',type(feature_extractor))

for item in feature_extractor:
    print(item)