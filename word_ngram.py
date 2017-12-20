

X_train_o = ["December 20th, 2017. We are at Starbucks working on the depression project. I don't know what I'm writing."]
for n in range(1,3):
    vect = CountVectorizer(tokenizer=lambda doc: make_ngrams(doc,n), min_df=1)
   
    X_train = vect.fit_transform(X_train_o)
  #  X_test = vect.transform(X_test_o)
  #  print('X_train_charngram.shape:\n{}'.format(X_train.shape))


    # To check features (the first 100)
    #feature_names = np.array(vect.get_feature_names())
    #print(feature_names[1000:1500])

 #   print("samples per class: {}".format(np.bincount(y_train)))
  #  print("Data: {}".format(np.bincount(y_test)))
    #print(X_train.shape)
  #  print('X_train.shape:\n{}'.format(X_train.shape))
    print(DataFrame(X_train.A, columns=vect.get_feature_names()).to_string())
