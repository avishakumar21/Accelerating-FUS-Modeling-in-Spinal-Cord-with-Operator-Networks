from nni.experiment import Experiment

if __name__=='__main__':
    
    search_space = {
        'batch': {'_type': 'choice', '_value': [2, 4, 8, 16,32]},
        'lr': {'_type': 'loguniform', '_value': [0.0001, 0.001]},
        'epochs': {'_type': 'choice', '_value': [ 750, 1000]} 
    }
    

    exp = Experiment('local')
    exp.config.trial_concurrency = 1
    exp.config.max_trial_number = 50
    exp.config.search_space = search_space
    exp.config.trial_command = 'python main.py'
    exp.config.trial_code_directory = '.'
    exp.config.tuner.name = 'GridSearch' #'TPE'
    exp.config.tuner.class_args = {'optimize_mode': 'minimize'}

    exp.run(8082)

    input("Press Enter to continue...")
    exp.stop()

    # commented out save model in main.py 
        # Store model
        # store_model(self.model,self.optimizer, epoch, self.result_folder)

    # changed prediction plot to false in config.json 