"""Calculate the runtime of BERT models"""
from adv_transformer.core.utils.data_loader import DataLoader
from adv_transformer.core.models.model import ClaimSpotterModel
import timeit
import numpy as np
import tensorflow as tf
from adv_transformer.core.utils.flags import FLAGS

def main():
    data_load = DataLoader()
    all_data = data_load.load_crossval_data()
    samples = np.array(all_data.x)[:100]
    samples = tf.data.Dataset.from_tensor_slices(
        ([x[0] for x in samples], [x[1] for x in samples])
    ).batch(FLAGS.cs_batch_size_reg)
    model = ClaimSpotterModel()
    model.warm_up()

    def run_inference_on_100_samples():
        all_preds = []
        for x_id, x_sent in samples:
            all_preds = all_preds + model.preds_on_batch((x_id, x_sent)).numpy().tolist()
        return all_preds

    print(timeit.timeit(run_inference_on_100_samples, number=10))

    


if __name__ == "__main__":
    main()