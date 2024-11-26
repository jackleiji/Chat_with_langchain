# Sweetalk Design Pattern

&emsp;&emsp;本项目主要基于《大话设计模式》这本书，对该书的设计原则和23个设计模式进行解读，原书是C#语言编写，本项目使用其他各种编程语言进行代码重写，供大家了解其他语言在设计模式中的编程方式和技巧。

&emsp;&emsp;《大话设计模式》这本书，通过各种生活中的例子，在小菜和大鸟的不断提问与回答过程中，对程序的不断重构演变，学习设计模式在案例中的适用性，更进一步掌握设计模式的异同和关键点。

## 使用说明

&emsp;&emsp;结合《大话设计模式》这本书，总结了各种设计模式的基本概念、知识点和适用场景，并使用各种编程语言进行代码重写。先通过书中的案例，了解23种设计模式及其相关的代码示例，再尝试用其他语言进行实现，如果遇到难以实现的设计模式，再来查看本项目中的代码示例。

&emsp;&emsp;如果觉得本项目中有错误，可以[点击这里](https://github.com/datawhalechina/sweetalk-design-pattern/issues)提交你希望补充的内容或者想要实现的编程语言，我们看到后会尽快进行补充。

### 在线阅读地址

https://datawhalechina.github.io/sweetalk-design-pattern/#/

### 进度安排

| 章节     | 内容                                                         | 负责人       |
| -------- | ------------------------------------------------------------ | ------------ |
| 前言     | 简介、内容概览（关联图）、设计理念                           | 长琴         |
| 设计原则 | 单一职责原则                                                 | 肖桐         |
|          | 开闭原则                                                     | 长琴         |
|          | 依赖倒置原则                                                 | 胡锐锋       |
|          | 迪米特原则                                                   | 碧涵         |
|          | 里氏替换原则                                                 | 长琴         |
| 设计模式 | 简单工厂模式、策略模式、装饰模式、代理模式、工厂方法模式     | 肖桐         |
|          | 原型模式、模板方法模式、外观模式、建造者模式                 | 碧涵         |
|          | 观察者模式、抽象工厂模式、状态模式、适配器模式               | 长琴         |
|          | 备忘录模式、组合模式、迭代器模式、单例模式                   | 胡锐锋       |
|          | 桥接模式、命令模式、职责链模式、中介者模式、享元模式、解释器模式、访问者模式 | 鸿飞         |
| 应用代码 | Java代码部分                                                 | 碧涵         |
|          | Python代码部分                                               | 肖桐         |
|          | C++代码部分                                                  | 长琴、胡锐锋 |



## 项目目录

<pre>
docs-----------------------------------------------大话设计模式
src------------------------------------------------示例代码
|   +---design_patterns--------------------------------设计模式示例代码
|   |   +---cpp--------------------------------------------C++语言示例代码
|   |   |   +---abstract_factory-------------------------------抽象工厂模式
|   |   |   +---adapter----------------------------------------适配器模式
|   |   |   +---bridge-----------------------------------------桥接模式
|   |   |   +---builder----------------------------------------建造者模式
|   |   |   +---chain_of_responsibility------------------------职责链模式
|   |   |   +---command----------------------------------------命令模式
|   |   |   +---composite--------------------------------------组合模式
|   |   |   +---decorator--------------------------------------装饰模式
|   |   |   +---facade-----------------------------------------外观模式
|   |   |   +---factory_method---------------------------------工厂方法模式
|   |   |   +---flyweight--------------------------------------享元模式
|   |   |   +---interpreter------------------------------------解释器模式
|   |   |   +---iterator---------------------------------------迭代器模式
|   |   |   +---mediator---------------------------------------中介者模式
|   |   |   +---memento----------------------------------------备忘录模式
|   |   |   +---observer---------------------------------------观察者模式
|   |   |   +---prototype--------------------------------------原型模式
|   |   |   +---proxy------------------------------------------代理模式
|   |   |   +---simple_factory---------------------------------简单工厂模式
|   |   |   +---singleton--------------------------------------单例模式
|   |   |   +---state------------------------------------------状态模式
|   |   |   +---strategy---------------------------------------策略模式
|   |   |   +---template_method--------------------------------模板方法模式
|   |   |   +---visitor----------------------------------------访问者模式
|   |   +---java-------------------------------------------Java语言示例代码
|   |   +---python-----------------------------------------Python语言示例代码
README.md------------------------------------------项目说明
</pre>


### 参考书籍

- GoF
- 深入设计模式
- 精通 Python 设计模式（第 2 版）
- 人人都懂设计模式：从生活中领悟设计模式
- 秒懂设计模式
- Python 设计模式（第 2 版）
- 设计模式之禅
- HeadFirst 设计模式
- [设计模式 | 菜鸟教程](https://www.runoob.com/design-pattern/design-pattern-tutorial.html)

### ChangeLog

- v1.2完成初版 221023
- v1.1完成笔记 221008
- v1.0基础结构 220916

## 关注我们
<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="resources/qrcode.jpeg" width = "180" height = "180">
</div>
&emsp;&emsp;Datawhale，一个专注于AI领域的学习圈子。初衷是for the learner，和学习者一起成长。目前加入学习社群的人数已经数千人，组织了机器学习，深度学习，数据分析，数据挖掘，爬虫，编程，统计学，Mysql，数据竞赛等多个领域的内容学习，微信搜索公众号Datawhale可以加入我们。

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。