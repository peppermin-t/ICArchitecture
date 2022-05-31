# -*-coding: utf-8-*-
import numpy as np
from bram import BRAM, BramConfig

class Matmul(object):
    '''矩阵乘法
        Args: uint8, (m, n)
        Args: int8, (n, p)
    '''

    def __init__(self):
        self.systolic_size = 4 # 脉动阵列大小
        self.bram = BRAM()
        pass

    def __call__(self, input: np.uint8, weight: np.int8):
        self.send_data(input, 'input')
        self.send_data(weight, 'weight')
        
        m, n = input.shape
        n, p = weight.shape
        self.send_instr(m, p, n)
        self.send_flag()
        self.wait_flag()
        output_arr = self.recv_output((m, p))
        return output_arr

    def send_data(self, data, block_name, offset='default'):
        '''写入input或weight至bram

            假设两个矩阵分别是(m,n) x (n,p), m和p的维度需要补全至self.systolic_size的倍数，
            并且写入时需要按照补零的方向写入，例如：  
                1. 矩阵(m, n)是m补零，则m个m个写入BRAM中。（行方向补零，列方向写入）  
                2. 矩阵(n, p)是p补零，则p个p个写入BRAM中。（列方向补零，行方向写入）
            
            Args:
                data: 要写入的数据
                block_name: input, weight
                offset: 偏移地址名称，默认为default
        '''
        row, col = data.shape
        pad = 0
        print(block_name + " Matrix")
        if(block_name == 'input'):
            pad = self.computePad(row)
            data = np.pad(data, ((0, pad), (0, 0)), 'constant').T # input send data in colomn dimension
        else:
            pad = self.computePad(col)
            data = np.pad(data, ((0, 0), (0, pad)), 'constant')
        
        self.bram.write(data, block_name)
        print(data)
            
        pass

    def send_instr(self, m, p, n):
        '''构建并发送指令

            两个矩阵shape分别为(m,n) x (n,p)
        '''
        ir = 0
        ir <<= 16
        ir += n
        ir <<= 16
        ir += p
        ir <<= 16
        ir += m
        print(ir)
        ir = ir.to_bytes(8, byteorder='little', signed=False)
        print("ir: " + str(ir))
        self.bram.write(ir, 'ir', 'instr')
        pass

    def send_flag(self):
        '''发送flag=1信号'''
        flag = b"\x01\x00\x00\x00"
        self.bram.write(flag, 'ir', offset='flag')
        pass
        
    def recv_output(self, output_shape: tuple):
        '''接收结果

            Args:
                output_shape: 输出的shape，类型tuple

            Return:
                output_arr: shape为output_shape的np.ndarray
        '''
        row, col = output_shape
        output_arr = self.bram.read(row * col * 4, 'output', dtype=np.int32).reshape(row, col)
        
        return output_arr
    
    def computePad(self, dim):
        left = dim % self.systolic_size
        if(left == 0):
            return 0
        return self.systolic_size - left
    
    def read_flag(self)->int:
        flag = self.bram.read( 1 , block_name = "ir" , offset = 'flag' )[0]
        return flag

    def wait_flag(self):
            value = -1
            while( value != 0):
                value = self.read_flag()
            print("GOT FLAG 0!")

if __name__ == '__main__':
    matmul = Matmul()
    

    ############ matrix 1
    x = np.random.randint(0, 2, (4,8), dtype=np.uint8)
    x = np.uint8(x)
    w = np.random.randint(-1, 2, (8,4), dtype=np.int8)

    std_output = np.matmul(x, w)
    print("my output")
    output = matmul(x, w)
    print(output)

    # err = output - std_output
    assert (output == std_output).all(), 'error'
    print('~~~ demo1 pass ~~~')
    

    ############ matrix 2
    x = np.random.randint(0, 5, (15,20), dtype=np.uint8)
    w = np.random.randint(-5, 5, (20,10), dtype=np.int8)

    std_output = np.matmul( x , w )
    
    output = matmul(x, w)

    # err = output - std_output
    assert (output == std_output).all(), 'error'
    print('~~~ demo2 pass ~~~')
