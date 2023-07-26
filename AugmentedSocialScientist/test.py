from AugmentedSocialScientist import bert, camembert
import pandas as pd
import numpy as np
"""
pd.options.display.max_colwidth=None
pd.options.display.max_rows=100

cb_train = pd.read_csv('../datasets/english/clickbait_train.csv')
cb_test = pd.read_csv('../datasets/english/clickbait_test.csv')
bert = bert.Bert()
train_loader = bert.encode(cb_train.headline.values, cb_train.is_clickbait.values)
test_loader = bert.encode(cb_test.headline.values, cb_test.is_clickbait.values)

score = bert.run_training(train_loader,
                          test_loader,
                          n_epochs=1,
                          lr=5e-5,
                          random_state=42,
                          save_model_as='clickbait')

cb_pred = pd.read_csv('../datasets/english/clickbait_pred.csv')
pred_loader = bert.encode(cb_pred.headline.values)
pred_proba = bert.predict_with_model(pred_loader, model_path='./models/clickbait')

cb_pred['pred_label'] = np.argmax(pred_proba, axis=1)
cb_pred['pred_proba'] = np.max(pred_proba, axis=1)

for i in range(20):
    print(f"{cb_pred.loc[i,'headline']}")
    print(f"Is clickbait: {bool(cb_pred.loc[i,'pred_label'])}, with a probability of {cb_pred.loc[i,'pred_proba']*100:.0f}%")
    print()
"""

off_train = pd.read_csv('../datasets/french/off_train.csv')
off_test = pd.read_csv('../datasets/french/off_test.csv')
camembert = camembert.CamemBert()
train_loader = camembert.encode(off_train.sentence.values, off_train.contains_off.values)
test_loader = camembert.encode(off_test.sentence.values, off_test.contains_off.values)

score = camembert.run_training(train_loader,
                               test_loader,
                               n_epochs=1,
                               lr=5e-5,
                               random_state=42,
                               save_model_as='off')

off_pred = pd.read_csv('../datasets/french/off_pred.csv')
pred_loader = camembert.encode(off_pred.sentence.values)
pred_proba = camembert.predict_with_model(pred_loader, model_path='./models/off')
off_pred['pred_label'] = np.argmax(pred_proba, axis=1)
off_pred['pred_proba'] = np.max(pred_proba, axis=1)