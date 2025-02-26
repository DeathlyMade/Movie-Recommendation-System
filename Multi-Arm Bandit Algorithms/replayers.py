import numpy as np
from tqdm import tqdm

class ReplaySimulator(object):
    '''
    A class to provide base functionality for simulating the replayer method for online algorithms.
    '''

    def __init__(self, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1, random_seed=1):

        np.random.seed(random_seed)
    
        self.reward_history = reward_history
        self.item_col_name = item_col_name
        self.visitor_col_name = visitor_col_name
        self.reward_col_name = reward_col_name

        # number of visits to replay/simulate
        self.n_visits = n_visits
        
        # number of runs to average over
        self.n_iterations = n_iterations
    
        # items under test
        self.items = self.reward_history[self.item_col_name].unique()
        self.n_items = len(self.items)
        
        # visitors in the historical reward_history (e.g., from ratings df)
        self.visitors = self.reward_history[self.visitor_col_name].unique()
        self.n_visitors = len(self.visitors)

        self.features = self.reward_history['combined_features'].unique()
        self.n_features = len(self.features)

    def reset(self):
        # number of times each item has been sampled (previously n_sampled)
        self.n_item_samples = np.zeros(self.n_items)
        
        # fraction of time each item has resulted in a reward (previously movie_clicks)
        self.n_item_rewards = np.zeros(self.n_items)
        
    
    def replay(self):
        
        results = []

        for iteration in tqdm(range(0, self.n_iterations)):
        
            self.reset()
            
            total_rewards = 0
            fraction_relevant = np.zeros(self.n_visits)

            for visit in range(0, self.n_visits):
            
                found_match = False
                while not found_match:
                
                    # choose a random visitor
                    visitor_idx = np.random.randint(self.n_visitors)
                    visitor_id = self.visitors[visitor_idx]

                    # select an item to offer the visitor
                    item_idx = self.select_item(visit)
                    item_id = self.items[item_idx]
                    
                    # if this interaction exists in the history, count it
                    reward = self.reward_history.query(
                        '{} == @item_id and {} == @visitor_id'.format(self.item_col_name, self.visitor_col_name))[self.reward_col_name]
                    
                    found_match = reward.shape[0] > 0
                
                reward_value = reward.iloc[0]
                
                self.record_result(visit, item_idx, reward_value)
                
                ## record metrics
                total_rewards += reward_value
                fraction_relevant[visit] = total_rewards * 1. / (visit + 1)
                
                result = {}
                result['iteration'] = iteration
                result['visit'] = visit
                result['item_id'] = item_id
                result['visitor_id'] = visitor_id
                result['reward'] = reward_value
                result['total_reward'] = total_rewards
                result['fraction_relevant'] = total_rewards * 1. / (visit + 1)
                
                results.append(result)
        
        return results
        
    def select_item(self, visit):
        return np.random.randint(self.n_items)
        
    def record_result(self, visit, item_idx, reward):
    
        self.n_item_samples[item_idx] += 1
        
        alpha = 1./self.n_item_samples[item_idx]
        self.n_item_rewards[item_idx] += alpha * (reward - self.n_item_rewards[item_idx])


class ABTestReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on an A/B test.
    '''
    
    def __init__(self, n_visits, n_test_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(ABTestReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
        
        # TODO: validate that n_test_visits <= n_visits
    
        self.n_test_visits = n_test_visits
        
        self.is_testing = True
        self.best_item_id = None
        
    def reset(self):
        super(ABTestReplayer, self).reset()
        
        self.is_testing = True
        self.best_item_idx = None
    
    def select_item(self, visit):
        if self.is_testing:
            return super(ABTestReplayer, self).select_item()
        else:
            return self.best_item_idx
            
    def record_result(self, visit, item_idx, reward):
        super(ABTestReplayer, self).record_result(visit, item_idx, reward)
        
        if (visit == self.n_test_visits - 1): # this was the last visit during the testing phase
            
            self.is_testing = False
            self.best_item_idx = np.argmax(self.n_item_rewards)
        

class EpsilonGreedyReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on an epsilon-Greedy bandit algorithm.
    '''

    def __init__(self, epsilon, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(EpsilonGreedyReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
    
        # parameter to control exploration vs exploitation
        self.epsilon = epsilon
    
    def select_item(self, visit):
        
        # decide to explore or exploit
        if np.random.uniform() < self.epsilon: # explore
            item_id = super(EpsilonGreedyReplayer, self).select_item()
            
        else: # exploit
            item_id = np.argmax(self.n_item_rewards)
            
        return item_id
    

class ThompsonSamplingReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on a Thompson Sampling bandit algorithm.
    '''

    def reset(self):
        self.alphas = np.ones(self.n_items)
        self.betas = np.ones(self.n_items)

    def select_item(self, visit):
    
        samples = [np.random.beta(a,b) for a,b in zip(self.alphas, self.betas)]
        
        return np.argmax(samples)

    def record_result(self, visit, item_idx, reward):
        
        ## update value estimate
        if reward == 1:
            self.alphas[item_idx] += 1
        else:
            self.betas[item_idx] += 1

class UCBReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on a UCB bandit algorithm.
    '''

    def __init__(self, c, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(UCBReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
    
        # parameter to control exploration vs exploitation
        self.c = c
        self.counter = 1
    def select_item(self, visit):
        self.counter = visit + 1
        # decide to explore or exploit
        ucb_values = [self.n_item_rewards[item_idx] + self.c * np.sqrt(np.log(self.counter) / (self.n_item_samples[item_idx] + 1)) for item_idx in range(0, self.n_items)]
        
        return np.argmax(ucb_values)

class LinUCBReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on a LinUCB bandit algorithm.
    '''
    
    def __init__(self, alpha, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(LinUCBReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
        
        self.alpha = alpha
        
        # number of features
        self.n_features = 10
        
        # initialize A and b
        self.A = [np.eye(self.n_features) for _ in range(0, self.n_items)]
        self.b = [np.zeros(self.n_features) for _ in range(0, self.n_items)]
    
    def select_item(self, visit):
        theta = [np.matmul(np.linalg.inv(self.A[item_idx]), self.b[item_idx]) for item_idx in range(0, self.n_items)]
        p = [np.matmul(np.array(theta[item_idx]).T, np.array(self.features[item_idx])) + self.alpha * np.sqrt(np.matmul(np.matmul(np.array(self.features[item_idx]).T, np.linalg.inv(self.A[item_idx])), np.array(self.features[item_idx]))) for item_idx in range(0, self.n_items)]
        
        return np.argmax(p)
    
    def record_result(self, visit, item_idx, reward):
        
        features = np.array(self.features[item_idx])
        
        self.A[item_idx] += np.matmul(features, features.T)
        self.b[item_idx] += reward * features

class GradientBanditReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on a Gradient Bandit algorithm.
    '''
    
    def __init__(self, alpha, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(GradientBanditReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
        
        self.alpha = alpha
        
        self.H = np.zeros(self.n_items)
        self.pi = np.zeros(self.n_items)
    
    def select_item(self, visit):
        
        self.pi = np.exp(self.H) / np.sum(np.exp(self.H))
        return np.random.choice(self.n_items, p=self.pi)
    
    def record_result(self, visit, item_idx, reward):
        
        for idx in range(0, self.n_items):
            if idx == item_idx:
                self.H[idx] += self.alpha * reward * (1 - self.pi[idx])
            else:
                self.H[idx] -= self.alpha * reward * self.pi[idx]
