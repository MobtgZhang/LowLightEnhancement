# include"source.hpp"

int main(int argc, char** argv){
    if (argc != 4){
        fprintf(stderr, "Usage: %s \t[input video]\t[output video]\t[threads]\n", argv[0]);
        return -1;
    }

    std::string load_video_filename = std::string(argv[1]);
    std::string save_video_filename = std::string(argv[2]);
    int num_threads = atoi(argv[3]);
    std::cout<<"load_video_filename: "<<load_video_filename<<std::endl;
    std::cout<<"save_video_filename: "<<save_video_filename<<std::endl;
    std::cout<<"num_threads: "<<num_threads<<std::endl;

	g_cap.open(load_video_filename);

	frams = static_cast<int>(g_cap.get(cv::CAP_PROP_FRAME_COUNT));     //视频帧数
	int frame_width = static_cast<int>(g_cap.get(cv::CAP_PROP_FRAME_WIDTH));  //视频帧宽度
	int frame_height = static_cast<int>(g_cap.get(cv::CAP_PROP_FRAME_HEIGHT)); //视频帧长度
	std::cout << "video has " << frams << " frams" << " wideth tmpw " << frame_width << " height " << frame_height << std::endl;//显示视频信息
	
	cv::VideoWriter writer(save_video_filename, cv::VideoWriter::fourcc('M','J','P','G'), 25, cv::Size(frame_width, frame_height));
	
	int nframs = frams;//用于计算视频处理速度的变量。

	int saved_frams  = frams;

	std::ofstream file("fps.txt"); // 打开文件（如果不存在则创建）
    
    if (!file.is_open()) { // 确保文件成功打开
		std::cout << "无法打开文件" << std::endl;
		return -1;
    }

	clock_t start, finish,tempory;
	int tmp_frames = 0;
	cv::Mat frame;
	if(num_threads>1){
		for(int k=0;k<2*num_threads;k++){
			std::vector<unsigned int> temp1;
			std::vector<unsigned int> temp2;
			for(int i=0;i<num_threads;i++){
				temp1.push_back(0);
				temp2.push_back(0);
			}
			task_index.push_back(temp1);
			pending_index.push_back(temp2);
		}
		for(int k=0;k<num_threads;k++){
			std::vector<cv::Mat> temp1;
			std::vector<cv::Mat> temp2;
			for(int i=0;i<num_threads;i++){
				temp1.push_back(cv::Mat());
				temp2.push_back(cv::Mat());
			}
			proposed_frams.push_back(temp1);
			pending_frams.push_back(temp2);
		}

		distribution_frams(num_threads);//为线程预分配处理帧
		//启用线程
		std::vector<std::thread> threads_list;
		for(unsigned int k=0;k<num_threads;k++){
			std::thread temp_task(thread_task,k+1,num_threads);
			threads_list.push_back(std::move(temp_task));
			threads_list[k].detach();
		}
		start = clock();//从线程打开开始计时，统计程序运行速度。
		//按照视频顺序存储视频帧
		int index[2] = { 0,0 };
		while (saved_frams > 0){
			//检查是否当前需要存储的帧已经转化完成
			if (ready_to_save(saved_frams, index,num_threads)){
				frame = proposed_frams[index[0]][index[1]];
				tempory = clock();
				double tmpfps = tmp_frames / (static_cast<double>(tempory - start) / CLOCKS_PER_SEC);
				std::string formattedNum = cv::format("%.2f", tmpfps);
				cv::putText(frame,"FPS:"+ formattedNum, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, 8);
				file << tmpfps << std::endl;
				writer << frame;                  //将转化好的帧存入视频中去
				cv::imshow("output", frame);       //实时刷新已经转化好的帧
				task_index[2 * index[0]][index[1]] = 0;                         //将该帧对应空间标记为数据已取出
				cv::waitKey(1);                                                 //等待1ms，此处如果不等待1ms，转换后的视频图像不会自动刷新
				saved_frams--;                                                  //
				tmp_frames++;
			}
			distribution_frams(num_threads);
		}
	}else{
		start = clock();//从线程打开开始计时，统计程序运行速度。
		while (saved_frams > 0){
			g_cap >> frame;
			tempory = clock();
			double tmpfps = tmp_frames / (static_cast<double>(tempory - start) / CLOCKS_PER_SEC);
			std::string formattedNum = cv::format("%.2f", tmpfps);
			cv::Mat temp;
			frame.convertTo(frame, CV_32F, 1.0 / 255);
			multi_fusion(frame, temp, alpha, fai);
			temp.convertTo(temp, CV_8U, 255);
			cv::putText(temp,"FPS:"+ formattedNum, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, 8);
			file << tmpfps << std::endl;
			writer << temp;                  //将转化好的帧存入视频中去
			cv::imshow("output", temp);       //实时刷新已经转化好的帧
			cv::waitKey(1);                                                 //等待1ms，此处如果不等待1ms，转换后的视频图像不会自动刷新
			saved_frams--;                                                  //
			tmp_frames++;
		}
	}
	finish = clock();
	//显示程序运行时间
	std::cout << "totally used time is " << static_cast<double>(finish - start) / CLOCKS_PER_SEC << " S" << std::endl;
	std::cout << "proposed speed " << nframs / (static_cast<double>(finish - start) / CLOCKS_PER_SEC) << " fps" << std::endl;
	cv::destroyAllWindows();
	writer.release();
	g_cap.release();
	return 0;
}
