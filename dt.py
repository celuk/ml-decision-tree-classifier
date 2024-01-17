# BIL 470
# Odev 1
# Seyyid Hikmet Celik
# 181201047

class DecisionTreeClassifier:
    # baslangicta karar agaci derinligini belirleyen constructor
    # varsayilan karar agaci derinligi 5
    def __init__(self, max_depth=5):
        self.root = None
        self.max_depth = max_depth

    # veri setini gini algoritmasina gore egitir
    def fit(self, X, y):
        self.root = self.recursive_tree(X, y)
        return

    # verilere bagli tahmin yapar
    def predict(self, X_test):
        if isinstance(X_test[0], float):
            node = self.root
            while node.left:
                if X_test[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.flower
        
        y_test = list()
        for element in X_test:
            node = self.root
            while node.left:
                if element[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right

            y_test.append(node.flower)
        return y_test
    
    # agac derinligine gore gini algoritmasini uygular ve agaci insa eder
    def recursive_tree(self, X, y, depth = 0):
        best_gini = float("inf")
        for index in range(len(X[0])):
            # gruplari, siniflari, girdileri olusturur
            for for_value in X:
                left, right = list(), list()
                for i,row in enumerate(X):
                    if row[index] <= for_value[index]:
                        left.append((row, y[i]))
                    else:
                        right.append((row, y[i]))
                groups = [left, right]

                classes = list(set(y))
                
                entries = sum([len(group) for group in groups])
                
                gini_value = 0
        
                for group in groups:
                    local_gini = 1
                    group_entries = len(group)
                    if group_entries == 0:
                        continue
        
                    for flower in classes:
                        local_gini = local_gini - ([entry[-1] for entry in group].count(flower) / group_entries) ** 2
                    gini_value = gini_value + local_gini * group_entries / entries
                
                if gini_value == 0:
                    return index, for_value[index], gini_value, groups
                
                if gini_value < best_gini:
                    best_gini = gini_value
                    best_index = index
                    best_value = for_value[index]
                    
        left_y = [entry[1] for entry in left]
        right_y = [entry[1] for entry in right]
        left = [entry[0] for entry in left]
        right = [entry[0] for entry in right]

        if (best_gini == 0 and len(set(y)) == 1) or depth == self.max_depth:
            node = Node(X, y, best_gini)
            node.flower = max([(flower, y.count(flower)) for flower in set(y)], key=lambda x : x[1])[0]
            return node

        node = Node(X, y, best_gini)
        node.feature_index = best_index
        node.threshold = best_value
        node.flower = max([(flower, y.count(flower)) for flower in set(y)], key=lambda x : x[1])[0]
        node.left = self.recursive_tree(left, left_y, depth + 1)
        node.right = self.recursive_tree(right, right_y, depth + 1)
        return node

# isimizi kolaylastirmak icin ekstra bir Node sinifi 
class Node:
    def __init__(self, X, y, gini):
        self.left = None
        self.right = None
        self.flower = None
        self.feature_index = 0
        self.threshold = 0
        self.X = X
        self.y = y
        self.gini = gini
