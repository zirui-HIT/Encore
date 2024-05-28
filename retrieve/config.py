class parameters():

    prog_name = "retriever"

    # set up your own path here
    root_path = "./"
    output_path = "./retrieve/checkpoint/"
    cache_dir = "./retrieve/cache/"

    # the name of your result folder.
    model_save_name = "bert-base"

    train_file = root_path + "retrieve/dataset/input/train.json"
    valid_file = root_path + "retrieve/dataset/input/dev.json"
    test_file = root_path + "retrieve/dataset/input/test.json"

    # model choice: bert, roberta
    pretrained_model = "bert"
    model_size = "bert-base-uncased"

    # pretrained_model = "roberta"
    # model_size = "roberta-base"

    # train, test, or private
    # private: for testing private test data
    device = "cuda"
    mode = "train"
    resume_model_path = ""

    # to load the trained model in test time
    saved_model_path = output_path + \
        '$(date "+%m%d-%H%M")/saved_model/model.pt'
    build_summary = False

    option = "rand"
    neg_rate = 3
    topn = 10

    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512
    max_program_length = 100
    n_best_size = 20
    dropout_rate = 0.1

    batch_size = 32
    batch_size_test = 16
    epoch = 20
    learning_rate = 2e-5

    report = 300
    report_loss = 100
