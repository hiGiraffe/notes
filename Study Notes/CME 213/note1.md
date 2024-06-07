# C++学习笔记

## 头文件宏定义

```c++
#ifndef _MATRIX_HPP
#define _MATRIX_HPP
……
#endif
```

在大的软件工程里面，可能存在多个文件同时包含一个头文件。
这种写法可以在生成可执行文件时避免头文件的重定义。

## 模板

* 函数模板

  ``` c++
  template <typename type> ret-type func-name(parameter list)
  {
  // 函数的主体
  }
  ```

  例子

    ```c++
  template <typename T>
  inline T const& Max (T const& a, T const& b)
  {
  return a < b ? b:a;
  }
  //使用
  int i = 39;
  int j = 20;
  cout <<Max(i, j) << endl;
    ```

* 类模板

  ```c++
  template <class type> class class-name {
  .
  .
  .
  }
  ```

  例子

    ```c++
  template <class T>
  class Stack {
  private:
  vector<T> elems;     // 元素
  
  public:
  void push(T const&);  // 入栈
  void pop();               // 出栈
  T top() const;            // 返回栈顶元素
  bool empty() const{       // 如果为空则返回真。
  return elems.empty();
  }
  };
  
  template <class T>
  void Stack<T>::push (T const& elem)
  {
  // 追加传入元素的副本
  elems.push_back(elem);    
  }
  
  template <class T>
  void Stack<T>::pop ()
  {
  if (elems.empty()) {
  throw out_of_range("Stack<>::pop(): empty stack");
  }
  // 删除最后一个元素
  elems.pop_back();         
  }
  
  template <class T>
  T Stack<T>::top () const
  {
  if (elems.empty()) {
  throw out_of_range("Stack<>::top(): empty stack");
  }
  // 返回最后一个元素的副本
  return elems.back();      
  }
  //使用
  Stack<int> intStack
  intStack.push(7); 
  cout << intStack.top() <<endl; 
    ```

## 虚拟函数 virtual

* 虚函数  
  c++重写中涉及到

  ```c++
  base *p = new inheriter;
  ```

  假如使用类中的virtual函数，则是派生类inheriter中的函数。
  假如使用正常函数，则会使用基类vase的函数。

* 虚基类
  共享基类不出问题

  ```c++
  //间接基类A
  class A{
  protected:
  int m_a;
  };
  //直接基类B
  class B: virtual public A{  //虚继承
  protected:
  int m_b;
  };
  //直接基类C
  class C: virtual public A{  //虚继承
  protected:
  int m_c;
  };
  //派生类D
  class D: public B, public C{
  public:
  void seta(int a){ m_a = a; }  //正确
  void setb(int b){ m_b = b; }  //正确
  void setc(int c){ m_c = c; }  //正确
  void setd(int d){ m_d = d; }  //正确
  private:
  int m_d;
  };
  ```

  假如不适用虚基类，D中的seta会因为有B和C两个seta而矛盾报错。

* 纯虚函数
  有纯虚函数的基类只能被继承，而不能实例化，需要在派生类中实现。

  ```c++
  class Base {
  public:
  virtual void pureVirtualFunction() = 0;
  };
  
  class Derived : public Base {
  public:
  void pureVirtualFunction() override {
  // 实现纯虚函数
  // ...
  }
  };
  ```

## 友元函数

```c++
class Box{
    double a;
public:
    friend void printWidth(Box box);
};

void printWidth(Box box){
    cout << box.width <<endl;
}
```

通过这种方法可以访问Box类中的所有变量。

## 运算符重载

一些常见的运算符和它们在C++中的重载用途：

* Arithmetic Operators (算术运算符):
  +, -, *, /, % 等用于重载加、减、乘、除和取模运算符。
* Comparison Operators (比较运算符):
  ==, !=, <, >, <=, >= 等用于重载相等、不相等、小于、大于、小于等于和大于等于运算符。
* Assignment Operators (赋值运算符):
  =, +=, -=, *=, /=, %= 等用于重载赋值和复合赋值运算符。
* Increment and Decrement Operators (自增和自减运算符):
  ++, -- 用于重载前缀和后缀自增和自减运算符。
* Indexing Operator (索引运算符):
  [] 用于重载类对象的索引运算符，使其可以像数组一样访问对象的元素。
* Function Call Operator (函数调用运算符):
  () 用于重载函数调用运算符，使对象可以像函数一样被调用。
* Member Access Operators (成员访问运算符):
  -> 用于重载成员访问运算符，使对象可以像指针一样访问成员。
* Stream Insertion and Extraction Operators (流插入和提取运算符):
  <<, >> 用于重载流插入和提取运算符，使自定义类型可以通过流进行输入和输出。

```c++
// 正常
Box operator+(const Box&);
//类的非成员函数
Box operator+(const Box&, const Box&);
```

## std::all_of

```c++
bool all_students_passed(const std::vector <Student> &students,
                         double pass_threshold) {
    return std::all_of(students.begin(),
                       students.end(),
                       [pass_threshold](Student s) {
                           double hw = s.homework * HOMEWORK_WEIGHT;
                           double mt = s.midterm * MIDTERM_WEIGHT;
                           double fe = s.final_exam * FINAL_EXAM_WEIGHT;
                           return hw + mt + fe >= pass_threshold;
                       }
    );
}
```

[pass_threshold] 是 lambda 函数中的一个捕获列表（capture list），用于指定 lambda 函数所捕获的外部变量。Lambda 函数可以通过捕获列表捕获外部变量，并在函数体内使用这些变量。

在这个特定的 lambda 函数中，[pass_threshold] 指明了 lambda 函数捕获了名为 pass_threshold 的外部变量。这意味着 lambda 函数可以在其函数体内访问并使用 pass_threshold 这个外部变量。

Lambda 函数的捕获列表有两种方式：

* 按值捕获: [var1, var2, ...] - 按值捕获指定变量。Lambda 函数拷贝这些变量的值，可以在函数体内读取但不能修改这些值。
* 按引用捕获: [&var1, &var2, ...] - 按引用捕获指定变量。Lambda 函数通过引用访问这些变量，可以在函数体内读取和修改这些变量

## 高精度时间测量

```c++
#include <chrono>
using namespace std::chrono;
int main(){
    high_resolution_clock::time_point start, end;
    duration<double> delta;
    start = high_resolution_clock::now();
    ...
    end = high_resolution_clock::now();
    delta = duration_cast<duration<double>>(end - start);
}
```

## openMP并行求和

```c++
#pragma omp parallel for reduction(+:sum0) reduction(+:sum1)
for(uint i=0; i<v.size(); i++) {
        if (v[i] % 2 == 0) {
            sum0 += v[i];
        }
        else {
            sum1 += v[i];
        }
    }
```

## C++部分求和函数

```c++
template< class InputIt, class OutputIt, class BinaryOperation >
OutputIt partial_sum( InputIt first, InputIt last, OutputIt d_first, BinaryOperation binary_op );
```

- `first` 和 `last`：表示输入序列的迭代器范围。`first` 指向要进行部分求和的序列的起始位置，而 `last` 指向序列的末尾位置（不包括）。
- `d_first`：表示输出序列的起始位置的迭代器，用于存储部分和的结果。
- `binary_op`：表示一个二元操作符（binary operator），用于指定如何组合两个元素。这个操作符将被用于执行部分和的计算。通常情况下，可以使用 `std::plus` 作为二元操作符，表示使用加法操作。

函数会计算并存储部分和的结果，其中结果的每个元素是从输入序列的开头到相应位置的部分和。它是一个累积过程，例如，第一个元素是输入序列的第一个元素，第二个元素是前两个元素的和，第三个元素是前三个元素的和，以此类推。

例子：

```c++
std::partial_sum(globalHisto.begin(), 
                     --globalHisto.end(), 
                     ++globalHistoExScan.begin(), 
                     std::plus<uint>());
```

这里是因为题目要求所以--和++，达到平衡



## transform函数

```
std::transform(first1, last1, result, unary_op);
```

其中：

- `first1` 和 `last1` 表示输入范围的起始和结束位置。
- `result` 表示输出范围的起始位置。
- `unary_op` 是一个一元操作（一元函数或者函数对象），用于对输入范围中的每个元素执行操作，并将结果存储到输出范围中。

如果要进行二元操作（接受两个参数的操作），`std::transform` 还可以采用以下形式：

```
cppCopy code
std::transform(first1, last1, first2, result, binary_op);
```

其中：

- `first1` 和 `last1` 表示第一个输入范围的起始和结束位置。
- `first2` 表示第二个输入范围的起始位置。
- `result` 表示输出范围的起始位置。
- `binary_op` 是一个二元操作，接受两个参数，分别来自第一个和第二个输入范围，然后执行操作，并将结果存储到输出范围中。

存储平方操作的例子

```C++
// 使用 std::transform 对输入数组中的元素进行平方操作，将结果存储到输出数组中
std::transform(input_array.begin(), input_array.end(), output_array.begin(),[](int x) { return x * x; });
```

作业代码中

```
std::transform(input_array.begin(), input_array.begin() + array_size,
	output_array.begin(), [&num_iter](elem_type &constant) {
        elem_type z = 0;
        for (size_t it = 0; it < num_iter; it++) {
        	z = z * z + constant;
        }
        return z;
	});
```

这里从开头到末尾遍历input_array，对于每一个元素，迭代num_iter次`z = z * z + constant;`运算。
