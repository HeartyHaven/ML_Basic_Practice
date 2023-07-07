import numpy as np
#-----------------------------预处理相关函数---------------------------------
def prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0,normalize_data=True):
    #功能：数据预处理
    num_examples=data.shape[0]
    data_proceed=np.copy(data)
    #预处理
    features_mean=0
    features_deviation=0
    data_normalized=data_proceed
    if normalize_data:#要不要做标准化操作：默认是要做标准化
    #为什么要做标准化?因为如果结果分布范围非常广，变化非常快，那么会对结果带来比较大的扰动。
    #变化大，扰动也大，那么除以std结果就会有很强的约束
        (
            data_normalized,
            features_mean,
            features_deviation

        )=normalize(data_proceed)
        data_proceed=data_normalized
    #特征变换sinusoidal
    if sinusoid_degree>0:
        sinusoids=generate_sinusoids(data_normalized,sinusoid_degree)
        data_proceed=np.concatenate((data_proceed,sinusoids),axis=1)
    #特征变换polynomial
    if polynomial_degree>0:
        polynomials=generate_polynomials(data_normalized,polynomial_degree,normalize_data)
        data_proceed=np.concatenate((data_proceed,polynomials),axis=1)
    #加一列1
    # data_proceed=np.hstack((np.ones((num_examples,1)),data_proceed))
    #这句话是为了方便做加常数项来确定的，很可能会报错维度不相同，这里还没解决
    return data_proceed,features_mean,features_deviation

def generate_sinusoids(dataset,sinusoid_degree):
    """sin(x)"""
    num_examples=dataset.shape[0]
    sinusoids=np.empty(num_examples,0)
    for degree in range(1,sinusoid_degree+1):
        sinusoid_features=np.sin(degree*dataset)
        sinusoids=np.concatenate((sinusoids,sinusoid_features),axis=1)
    return sinusoids
"""
这段代码的作用是生成一组正弦函数特征，用于对输入数据进行非线性变换。
具体来说，它将给定的数据集按照每个特征生成一组正弦函数特征，其中每个正弦函数的频率是该特征的度数（degree）。
最终返回一个新的数据集，其中每个样本都被表示为一组正弦函数特征。
这种特征变换的目的是提高拟合的效果，相当于变成了非线性函数关系，变量都套皮了sin()。这可能参考了自然界信号和
傅里叶变换提到的三角函数可以定义一切函数的思想。
"""
def generate_polynomials(dataset,polynomial_degree,normalize_data):
    """变换方法：比如两个属性x1,x2，经过处理后又会生成：x1^2,x2^2,x1x2，甚至更多"""
    features_split=np.array_split(dataset,2,axis=1)
    #这个函数的作用是分隔数组，2表示分成2个子数组，axis=1表示用轴1也就是y轴分隔。这里默认是2个属性，所以分成两个单属性数组
    dataset1=features_split[0]
    dataset2=features_split[2]
    (num_examples1,numfeatures1)=dataset1.shape
    (num_examples2,numfeatures2)=dataset2.shape#分别获取样例数和特征数
    if num_examples1!=num_examples2:
        raise ValueError("无法对两个样本数不同的数组进行多重非线性化")
    if numfeatures1==0 or numfeatures2==0:
        raise ValueError("无法对空的内容做多重非线性化")
    if num_examples1==0:
        dataset1=dataset2#可以处理是因为这样得到的结果就是单变量表达式
    elif num_examples2==0:
        dataset2=dataset1
    numfeatures=numfeatures1 if numfeatures1<numfeatures2 else numfeatures2#二者取小作为特征数，多出的特征舍弃
    dataset1=dataset1[:,numfeatures]
    dataset2=dataset2[:,numfeatures]#获取相关的属性值
    polynomials=np.empty((num_examples1,0))#里面放的元组是一个参数，代表shape，创建一个有完整行数但无列数的空数组
    #目的可能是为了在后续的循环中将多个多项式特征拼接成一个矩阵
    for i in range(1,polynomial_degree+1):
        for j in range(1+1):
            polynomial_feature=(dataset1**(i-j))**(dataset2**j)
            polynomials=np.concatenate((polynomials,polynomial_feature),axis=1)#相当于是把两列拼在了一起
        if normalize_data:
            polynomials=normalize(polynomials)[0]#如果指定了正则化，那么生成其他变量的时候还需要做一下正则化
    return polynomials
    #这个函数的作用是从2个变量多项式地生成多变量，生成最高次数是多少是由你的polynomial_degree决定的。
    #具体的生成方法就是利用循环迭代产生新的变量组合，而且默认把原来的特征组等分成两部分，看作是两个变量。
    #它的作用就是生成更加平滑的非线性关系，来提高拟合效果

def normalize(features):#正态分布的归一化公式：x->(x-μ)/θ，其中μ是均值，θ是方差
    features_normalized=np.copy(features).astype(float)
    # features_normalized[np.isnan(features_normalized)] = 0
    # features0=features_normalized.ravel()

    features_mean=np.mean(features_normalized,0)           #计算均值
    features_deviation=np.std(features_normalized,0)       #计算标准差
    #标准化操作：所有的数据都映射到更小的区间内
    if features.shape[0]>1:#有不止一条数据
        features_normalized-=features_mean
    #防止除以0
    features_deviation[features_deviation==0]=1#方差为0的地方改变为1。这个写法挺新的，这个变量代表自己也代表他的后代（用了同一个变量名）
    features_normalized/=features_deviation#然后缩小区间范围到(-1,1)之间
    return features_normalized,features_mean,features_deviation

#这两个函数都是做预处理用的，但是我没有从视频里扒出来。所以目前上面函数的主要作用就是最后加了一列1，然后算了一点数字
#------------------------定义线性回归类---------------------------
class LinearRegression:
    def __init__(self,data,labels,polynomial_degree=0,sinusoid_degree=0,normalize_data=True):
        """
        这个函数的作用就是计算数据的一些属性值并且返回经过预处理的数据：预处理是把数据放缩到一个相对小的范围内
        然后得到特征的个数
        初始化参数矩阵
        """
        (data_proceed,#预处理完毕之后的数据
            features_mean,#预处理完毕之后的数据的均值和方差
            features_deviation)=prepare_for_training(data,
                                                    polynomial_degree=0,
                                                    sinusoid_degree=0,
                                                    normalize_data=True)
        self.data=data_proceed
        self.labels=labels
        self.features_mean=features_mean
        self.features_deviation=features_deviation
        self.polynomial_degree=polynomial_degree
        self.sinusoid_degree=sinusoid_degree
        self.normalize_data=normalize_data
        num_features=data.shape[1]#维度数==列数
        self.num_features=num_features
        # print(num_features)
        self.theta=np.zeros((num_features,1))#构造权重参数矩阵，其实就是一个0向量。
        pass
    
    def train(self,alpha,num_iterations=100):
        cost_history=self.gredient_descent(alpha,num_iterations)#不仅更新参数theta，还要记录。
        return self.theta,cost_history

    def gredient_descent(self,alpha,num_iterations):
        cost_history=[]
        for i in range(num_iterations):
            self.gredient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
            #这里是每经历一轮梯度下降之后计算一下损失函数做记录。
        return cost_history#这里就是做一个可视化用的
    
    def gredient_step(self,alpha):
        #单次梯度下降的实际操作。
        num_examples=self.data.shape[0]
        prediction=LinearRegression.hypothesis(self.data,self.theta)#得到了预测值
        delta=prediction-self.labels#得到了预测值与真实值之间的差距
        theta=self.theta            #用于更新的那个权重参数矩阵
        #套用小批量梯度下降的公式：
        theta=theta-alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        #参考了theta更新公式;同时num_examples的应用使得这个函数既可以做全局的梯度下降也可以小批量梯度下降。
        self.theta=theta
        #这里delta和data都是横着的向量，为了实现对应元素相乘
        pass

    """
      静态方法是一种在类中定义的方法，它与实例无关，因此可以在不创建类实例的情况下调用。
      与普通方法不同，静态方法没有self参数，因此它不能访问实例属性和方法。
      """
    @staticmethod
    def hypothesis(data,theta):
        #进行一次预测操作：就是样本点乘以参数矩阵。
        predictions=np.dot(data,theta)
        return predictions          
    
    def cost_function(self,data,labels):
        num_examples=data.shape[0]#总的样本个数
        delta=LinearRegression.hypothesis(self.data,self.theta)-labels#得到了预测值
        cost=(1/2)*np.dot(delta.T,delta)
        #这里要说一下np.dot的含义：就是常规的矩阵乘法，也就是竖着的*横着的，最后相加得到结果矩阵中的一个值
        return cost[0][0]#结果应该就是个1*1的矩阵，所以取0，0
    
#------------------------辅助函数-------------------------------------
#训练模块
    def get_cost(self,data,labels):
        #用已有的数据进行训练。
        #step1:数据预处理
        data_processed=prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0,normalize_data=True)[0]
        #我们只取预处理之后的函数
        return self.cost_function(data_processed,labels)#得到当前的损失

    def predict(self,data):
        #用训练好的参数做预测，返回回归值
        data_processed=prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0,normalize_data=True)[0]
        predictions=LinearRegression.hypothesis(data_processed,self.theta)
        return predictions