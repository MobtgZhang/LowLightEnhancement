#pragma once
#include<opencv2/opencv.hpp>
#include<string>
#include<iostream>
#include<thread>
# include"lime.hpp"

unsigned int free_space(unsigned int i,unsigned int nums_thread);
unsigned int ready_to_save(int saved_frams, int *index,unsigned int nums_thread);
void distribution_frams(unsigned int nums_thread);
unsigned int get_frams(unsigned int i, unsigned int *index,unsigned int nums_thread);

cv::VideoCapture g_cap;
int frams;

const double alpha{ 2 }, fai{ 4.3633 };
//参数window_1定义在头文件 get_illumination_map.h中，用以定义形态学运算窗口大小。

//处理后的帧缓冲区域

std::vector<std::vector<unsigned int> >task_index;//处理后的帧索引，包含了帧的顺序标号，以及缓存区的状态。
std::vector<std::vector<cv::Mat> >proposed_frams;//处理后的帧存储buff

std::vector<std::vector<unsigned int> >pending_index;
std::vector<std::vector<cv::Mat> >pending_frams;

void thread_task(unsigned int num,unsigned int nums_thread){
	cv::Mat frame;
	unsigned int save_area{ 0 };
	unsigned int index[2]{ 0,0 };
	while (true){
		save_area = free_space(num,nums_thread);
		if (save_area != 100)//如果缓冲区有空闲位置
		{
			if (get_frams(num, index,nums_thread))//查询待处理缓冲区是否有待处理帧
			{
				(pending_frams[index[0]][index[1]]).copyTo(frame);
				if (frame.empty()){
					std::cout << "empty frame in thread1" << std::endl;
				}
				task_index[2*num-1][save_area] = pending_index[2 * index[0] + 1][index[1]];//读取当前视频帧标号
				pending_index[2 * index[0]][index[1]] = 0;

				cv::Mat temp;
				frame.convertTo(frame, CV_32F, 1.0 / 255);
				multi_fusion(frame, temp, alpha, fai);
				temp.convertTo(temp, CV_8U, 255);
				//cv::vconcat(frame, temp, temp);

				proposed_frams[num-1][save_area] = temp;//将当前处理完成后的帧存放到帧缓冲区中
				task_index[2*(num-1)][save_area] = 1;
				std::cout << "thread "<<num<<" proposed fram " << task_index[num][save_area] << std::endl;
			}

		}
	}
}

//查找当前帧缓冲区是否有空闲位置，如果有返回空闲位置索引，没有返回标志100
//i  对应线程标号。
//j  空闲位置索引
unsigned int free_space(unsigned int i,unsigned int nums_thread){
	for (unsigned int j = 0; j < nums_thread; j++){
		if (task_index[(i - 1) * 2][j] == 0){
			return j;
		}
	}
	return 100;
}

//查找当前顺序下需要存储的帧是否处理完成。
//saved_frams  当前需要存储的帧标号
//index        用于返回当前需要存储的帧的位置
unsigned int ready_to_save(int saved_frams, int *index,unsigned int nums_thread)
{
	for (int i = 0; i < nums_thread; i++)
	{
		for (int j = 0; j < nums_thread; j++)
		{
			if (task_index[i * 2 + 1][j] == saved_frams)
			{
				if (task_index[i * 2][j] == 1)
				{
					index[0] = i;
					index[1] = j;
					return 1;
				}
			}
		}
	}
	return 0;// 当前需要存储帧尚未处理完成。
}


//向待处理帧缓冲区分配数据
void distribution_frams(unsigned int nums_thread)
{
	for (unsigned int j = 0; j < nums_thread; j++)
	{
		for (unsigned int i = 0; i < nums_thread; i++)
		{
			if (pending_index[i * 2][j] == 0)//查询是否有空余空间提供分配
			{
				if (frams > 0)
				{
					pending_index[(i * 2) + 1][j] = frams;//标记存储帧顺序
					g_cap >> pending_frams[i][j];       //将待处理帧流出到数组
					--frams;                            //更新当前剩余帧数
					pending_index[i * 2][j] = 1;        //更新索引，表示该缓冲区以及可以读取数据
				}
			}
		}
	}
}
//从待处理帧缓冲区中获取当前待处理帧
//i     表示线程标号
//index 用于返回代取帧在代取帧缓冲区中的索引。
unsigned int get_frams(unsigned int i, unsigned int *index,unsigned int nums_thread)
{
	i = i - 1;
	unsigned int max_frams{ 0 };
	for (unsigned int j = 0; j < nums_thread; j++)
	{
		if (pending_index[i * 2][j] == 1)
		{
			if (pending_index[(i * 2) + 1][j] > max_frams)
			{
				max_frams = pending_index[i * 2 + 1][j];
				index[0] = i;
				index[1] = j;
			}
		}
	}
	return max_frams;
}
