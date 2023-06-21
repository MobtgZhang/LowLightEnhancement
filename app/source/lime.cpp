# include "lime.hpp"
void get_illuminationmap(const cv::Mat& image_double,const cv::Mat*HSV_channel,cv::Mat &I1,cv::Mat &I2,cv::Mat &I3,cv::Mat &reflect){
	//初步评估光照图
	cv::Mat I;
	//对开运算闭运算需要用到的原型模板，原型模板直径大小为 window_1
	cv::Matx<unsigned char, window_1, window_1> element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(window_1, window_1));
	cv::morphologyEx(HSV_channel[2], I, cv::MORPH_CLOSE, element);

	//0.013s
	I1 = fastGuidedFilter(HSV_channel[2], I, 15, 0.001, 1);//采用HSV_V进行导向滤波得到光照图I1
	//0.222s 这个时间正常
    //	cv::imshow("illumation map 1", I1);//显示光照图1
    //计算反射图，通过retinex理论，由原图按照元素除以光照图即可得到反射图
	//假设三个色彩通道的光照图一致，将光照图I1复制到三个通道进而计算反射图
	cv::Mat I1_3channel;
	std::vector<cv::Mat> illumation_map_temp;
	illumation_map_temp.push_back(I1);
	illumation_map_temp.push_back(I1);
	illumation_map_temp.push_back(I1);
	cv::merge(illumation_map_temp, I1_3channel);
	reflect = image_double / I1_3channel;
	//0.364s
	//计算光照图I2，光照图I2是为了增强全局光照让图像的暗部区域显示得更加清楚，作者采用了arctan变换来计算光照图I2
	double I_mean = mean(I1)[0];
	double lamuda = 10 + ((1 - I_mean) / I_mean);//lamuda是自适应参数，用于进行arctan变换时归一化效果
	I2 = lamuda * I1;
	I2 = (-1) / (I2 + 1) + 1;
	//0.385s
	//光照图I3是为了增强图像对比度，将光照图I1进行CLAHE变换后得到的
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(10, cv::Size(10, 10));//第一个参数为分块对比度限制，第二个参数为分块多少
	//OpenCV子带的直方图均衡化采用的是unsigned int 8类型的数据，此处需要将浮点型的光照图转换到uint8类型
	cv::Mat temp;
	I1.convertTo(temp, CV_8UC1, 255.0);
	clahe->apply(temp, I3);
	I3.convertTo(I3, CV_32F, 1.0 / 255);
}
//https ://blog.csdn.net/kuweicai/article/details/78385871 
//导向滤波的代码在CSDN得到，是何凯明在2015年发表的 fast guided filter中提出的算法
cv::Mat fastGuidedFilter(const cv::Mat &I_org, const cv::Mat &p_org, int r, double eps, int s){
	/*
	% GUIDEDFILTER   O(N) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	cv::Mat I, _I;
	I_org.convertTo(_I, CV_32FC1, 1.0);

	resize(_I, I,I.size(), 1.0 / s, 1.0 / s, 1);


	cv::Mat p, _p;
	p_org.convertTo(_p, CV_32FC1, 1.0);
	//p = _p;
	resize(_p, p, p.size(), 1.0 / s, 1.0 / s, 1);

	//[hei, wid] = size(I);    
	int hei = I.rows;
	int wid = I.cols;

	r = (2 * r + 1) / s + 1;//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4   

	//mean_I = boxfilter(I, r) ./ N;    
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_32FC1, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;    
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_32FC1, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;    
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, CV_32FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.    
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;    
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_32FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;    
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;       
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;    
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;    
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_32FC1, cv::Size(r, r));
	cv::Mat rmean_a;
	resize(mean_a, rmean_a, I_org.size(), 1);

	//mean_b = boxfilter(b, r) ./ N;    
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_32FC1, cv::Size(r, r));
	cv::Mat rmean_b;
	resize(mean_b, rmean_b, I_org.size(), 1);

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;    
	cv::Mat q = rmean_a.mul(_I) + rmean_b;

	return q;
}


void cal_weight(const cv::Mat &image, const cv::Mat *HSV_channel,cv::Mat &weight_bright, cv::Mat &weight_constract, const double &alpha, const double &fai){
	//计算亮度权重 weight_bright=exp(-(image-0.5)^2/(2*0.25^2))  所有运算均为矩阵中元素的点运算非矩阵运算
	weight_bright = image - 0.5;
	weight_bright = weight_bright.mul(weight_bright);
	weight_bright = -weight_bright / 0.125;

	//对该部分采用幂函数拟合，将拟合后的结果代替上述结果
	cv::Mat wb_3, wb_2;
	cv::pow(weight_bright, 3, wb_3);
	cv::pow(weight_bright, 2, wb_2);
	weight_bright = (0.0571*wb_3) + (0.3699*wb_2) + (0.9395*weight_bright) + 0.9929;

	//根据对比度和色彩饱和度计算权重
	weight_constract = (alpha*(HSV_channel[0] / 360)) + fai;

	//对该部分采用幂函数拟合，用幂函数拟合结果代替余弦变换结果
	cv::pow(weight_constract, 3, wb_3);
	cv::pow(weight_constract, 2, wb_2);
	weight_constract = ((-0.1254)*wb_3) + (1.735*wb_2) - (7.002*weight_constract) + 8.58;
	weight_constract = weight_constract.mul(image);
	weight_constract = weight_constract.mul(HSV_channel[1]);
	cv::GaussianBlur(weight_constract, weight_constract, cv::Size(21, 21), 0, 0, cv::BORDER_REFLECT);
	weight_constract = cv::max(weight_constract, 0.000001);

}

void multi_fusion(const cv::Mat& image_double, cv::Mat &image_enhanced, const double & alpha, const double &fai){
	cv::Mat image_hsv;//原图在hsv域的矩阵，后续计算需要用到图像三个通道的最大值（HSV_V），以及计算对比度权重时需要用到。
	cv::cvtColor(image_double, image_hsv, CV_BGR2HSV);//转换到hsv域

	//存储HSV三个通道 从image_hsv中分离出来
	cv::Mat HSV_channel[3];
	//计算光照图
	cv::split(image_hsv, HSV_channel);//将image_hsv中的三个通道分离到HSV_channel中去

	cv::Mat I1, I2, I3, reflect;
	get_illuminationmap(image_double, HSV_channel, I1, I2, I3, reflect);


	//计算权重图
	//对三幅光照图分别从光照强度，bright，对比度 contract两个方面来计算对应的权重，然后将两者相乘得到每幅光照图对应的权重
	cv::Mat w1_b, w1_c;
	cal_weight(I1, HSV_channel, w1_b, w1_c, alpha, fai);
	cv::Mat w2_b, w2_c;
	cal_weight(I2, HSV_channel, w2_b, w2_c, alpha, fai);
	cv::Mat w3_b, w3_c;
	cal_weight(I3, HSV_channel, w3_b, w3_c, alpha, fai);

	cv::Mat w1, w2, w3;
	w1 = w1_b.mul(w1_c);
	w2 = w2_b.mul(w2_c);
	w3 = w3_b.mul(w3_c);

	//以三幅光照图的权重和作为分母对三幅光照图的权重进行加权平均
	cv::Mat w_sum;
	w_sum = w1 + w2 + w3;
	cv::Mat w1_final, w2_final, w3_final;
	w1_final = w1 / w_sum;
	w2_final = w2 / w_sum;
	w3_final = w3 / w_sum;
	//	cv::imshow("w1_final", w1_final);
	//	cv::imshow("w2_final", w2_final);
	//	cv::imshow("w3_final", w3_final);

		//计算总的光照图
	cv::Mat I_fusion;
	I_fusion = I1.mul(w1_final) + I2.mul(w2_final) + I3.mul(w3_final);

	//将最终的光照图复制到三个通道上，作为总的光照图
	std::vector<cv::Mat> I_fusion_3channel;
	I_fusion_3channel.push_back(I_fusion);
	I_fusion_3channel.push_back(I_fusion);
	I_fusion_3channel.push_back(I_fusion);
	cv::merge(I_fusion_3channel, I_fusion);

	//采用retinex理论计算最终增强后的图像结果
	image_enhanced = reflect.mul(I_fusion);
	cv::pow(image_enhanced, 0.75, image_enhanced);
}
