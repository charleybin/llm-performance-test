# 导入 concurrent.futures 模块，用于实现多线程并发执行任务
import concurrent.futures
# 导入 json 模块，用于处理 JSON 数据的编码和解码
import json
# 导入 time 模块，用于获取和处理时间相关的操作
import time
# 导入 csv 模块，用于读写 CSV 文件
import csv
# 导入 requests 模块，用于发送 HTTP 请求
import requests
# 导入 random 模块，用于随机选择提示词
import random
# 导入 tqdm 模块，用于显示进度条
from tqdm import tqdm
# 导入 datetime 模块，用于格式化日期和时间
import datetime

# 加载提示词文件
def load_prompts(prompts_file_path):
    """
    从指定路径加载提示词文件

    :param prompts_file_path: 提示词文件的路径
    :return: 包含提示词的列表
    """
    prompts = []
    # 先计算文件的行数
    with open(prompts_file_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    
    # 重新打开文件并使用tqdm显示进度
    with open(prompts_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=line_count, desc="加载提示词"):
            try:
                data = json.loads(line.strip())
                # 从每行JSON中提取提示词，使用 "question" 键
                if "question" in data:
                    prompts.append(data["question"])
            except json.JSONDecodeError:
                continue
    return prompts

# 加载配置文件
def load_config(config_path):
    """
    从指定路径加载配置文件

    :param config_path: 配置文件的路径
    :return: 解析后的配置文件内容
    """
    # 以只读模式打开配置文件，并指定编码为 UTF-8
    with open(config_path, "r", encoding="utf-8") as f:
        # 使用 json.load 函数解析配置文件中的 JSON 数据
        config = json.load(f)
    # 返回解析后的配置文件内容
    return config

# 初始化请求头
def initialize_headers(config):
    """
    根据配置文件初始化 HTTP 请求头

    :param config: 配置文件内容
    :return: 包含请求头信息的字典
    """
    # 如果配置中定义了自定义headers，则使用配置中的headers
    if "headers" in config:
        headers = config["headers"].copy()
        # 处理headers中可能需要替换的变量
        for key, value in headers.items():
            if isinstance(value, str) and "{api_key}" in value:
                headers[key] = value.replace("{api_key}", config["api_key"])
    else:
        # 否则使用默认的headers
        headers = {
            # 指定请求内容的类型为 JSON
            "Content-Type": "application/json",
            "x-bxy-id": f"{config['api_key']}",
            # 使用配置文件中的 API 密钥进行授权
            "Authorization": f"Bearer {config['api_key']}"
        }
    # 返回请求头字典
    return headers

# 请求函数，使用流式输出
def request_completion(headers, base_url, model, prompt, temperature, max_tokens, top_p):
    """
    发送请求并获取流式响应

    :param headers: HTTP 请求头
    :param base_url: 请求的基础 URL
    :param model: 使用的模型名称
    :param prompt: 输入的提示信息
    :param temperature: 控制生成文本的随机性
    :param max_tokens: 生成文本的最大 token 数
    :param top_p: 采样概率
    :return: 响应结果、响应耗时、首 token 延迟、非首 token 平均延迟、每秒生成 token 数、输入 token 数、总 token 数、输出 token 数、请求开始时间、请求结束时间
    """
    # 记录请求开始时间
    start_time = time.time()
    # 定义请求数据字典，包含模型、温度、最大 token 数、采样概率、提示信息和流式输出标志
    data = {
        # 指定使用的模型
        "model": model,
        # 设置生成文本的随机性
        "temperature": temperature,
        # 设置生成文本的最大 token 数
        "max_tokens": max_tokens,
        # 设置采样概率
        "top_p": top_p,
        # 输入的提示信息
        "messages": [{"role": "user", "content": prompt}],
        # 开启流式输出
        "stream": False
    }
    # 用于存储最终的响应文本
    response_text = ""
    # 用于存储所有的响应块
    full_response = []
    try:
        # 发送 POST 请求，并开启流式响应
        with requests.post(f"{base_url}/chat/completions", headers=headers, json=data, stream=True) as response:
            # 检查响应状态码，如果不是 200 则抛出异常
            response.raise_for_status()
            # 遍历响应的每一行
            for line in response.iter_lines():
                if line:
                    # 去除行首的 "data: " 并解码为字符串
                    line = line.lstrip(b'data: ').decode('utf-8')
                    if line == "[DONE]":
                        # 如果遇到结束标志，则跳出循环
                        break
                    try:
                        # 解析 JSON 数据
                        chunk = json.loads(line)
                        # 将解析后的块添加到 full_response 列表中
                        full_response.append(chunk)
                        # 获取响应块中的 delta 字段
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        # 获取 delta 字段中的 content 内容
                        content = delta.get('content', '')
                        # 将内容添加到响应文本中
                        response_text += content
                    except json.JSONDecodeError:
                        # 如果解析 JSON 数据出错，则跳过当前行
                        continue
    except requests.RequestException as e:
        # 打印请求出错信息
        print(f"请求出错: {e}")
    # 记录请求结束时间
    end_time = time.time()
    # 计算响应耗时并转换为毫秒
    latency = (end_time - start_time) * 1000
    # 注释提示接下来要解析完整的 JSON 数据
    # 解析完整的 JSON 数据
    # 初始化一个字典，用于合并所有响应块中的 delta 信息
    combined_response = {"choices": [{"delta": {}}]}
    # 初始化一个空字典，用于存储使用信息
    usage = {}
    # 遍历所有响应块
    for chunk in full_response:
        # 检查当前响应块中是否包含使用信息
        if 'usage' in chunk:
            # 如果包含，将使用信息更新到 usage 字典中
            usage = chunk['usage']
        # 检查choices是否存在且不为空
        if chunk.get('choices') and len(chunk['choices']) > 0:
            # 合并当前响应块中的 delta 信息到 combined_response 中
            combined_response['choices'][0]['delta'] = {
                # 保留已有的 delta 信息
                **combined_response['choices'][0]['delta'],
                # 合并新的 delta 信息，如果不存在则使用空字典
                **chunk.get('choices', [{}])[0].get('delta', {})
            }
    # 从 usage 字典中获取首 token 延迟，如果不存在则默认为 0
    first_token_latency = usage.get("time_to_first_token_ms", 0)
    # 从 usage 字典中获取非首 token 平均延迟，如果不存在则默认为 0
    non_first_token_avg_latency = usage.get("time_per_output_token_ms", 0)
    # 从 usage 字典中获取输入 token 数，如果不存在则默认为 0
    prompt_tokens = usage.get("prompt_tokens", 0)
    # 从 usage 字典中获取总 token 数，如果不存在则默认为 0
    total_tokens = usage.get("total_tokens", 0)
    # 从 usage 字典中获取输出 token 数，如果不存在则默认为 0
    completion_tokens = usage.get("completion_tokens", 0)
    # 从 usage 字典中获取每秒生成 token 数，如果不存在则默认为 0
    tokens_per_second = usage.get("tokens_per_second", total_tokens / latency * 1000)
    # 返回响应结果、响应耗时、首 token 延迟、非首 token 平均延迟、每秒生成 token 数、输入 token 数、总 token 数、输出 token 数、请求开始时间、请求结束时间
    return {"choices": [{"message": {"content": response_text}}]}, latency, first_token_latency, non_first_token_avg_latency, tokens_per_second, prompt_tokens, total_tokens, completion_tokens, start_time, end_time

# 多线程并发请求
def concurrent_requests(headers, config, prompts=None):
    """
    并发执行多个请求

    :param headers: HTTP 请求头
    :param config: 配置文件内容
    :param prompts: 提示词列表，如果提供则随机选择，否则使用配置中的默认提示词
    :return: 包含所有请求结果的列表
    """
    # 从配置文件中获取请求的基础 URL
    base_url = config["base_url"]
    # 初始化一个空列表，用于存储所有请求的结果
    results = []
    # 初始化一个空列表，用于存储所有的 future 对象
    futures = []
    # 创建一个线程池执行器，使用与任务数相同的工作线程数
    max_workers = config.get("max_workers", config["num_requests"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用tqdm创建进度条来展示任务提交进度
        for _ in tqdm(range(config["num_requests"]), desc="提交任务"):
            # 如果提供了提示词列表且不为空，则随机选择一个提示词
            prompt = random.choice(prompts) if prompts and len(prompts) > 0 else config.get("prompt_config", {}).get("default_prompt", "")
            # 提交请求任务到线程池
            future = executor.submit(
                request_completion,
                headers,
                base_url,
                config["model_name"],
                prompt,
                config["temperature"],
                config["max_tokens"],
                config["top_p"]
            )
            # 记录任务提交时间
            future.start_time = time.time()
            # 将 future 对象添加到 futures 列表中
            futures.append(future)

        # 使用tqdm显示任务完成进度
        completed_futures = []
        with tqdm(total=len(futures), desc="执行任务") as pbar:
            for future in concurrent.futures.as_completed(futures):
                # 获取任务的结果
                response, latency, first_token_latency, non_first_token_avg_latency, tokens_per_second, prompt_tokens, total_tokens, completion_tokens, start_time, end_time = future.result()
                # 计算等待时间
                wait_time = start_time - future.start_time
                # 将结果添加到 results 列表中
                results.append((response, latency, first_token_latency, non_first_token_avg_latency, tokens_per_second, prompt_tokens, total_tokens, completion_tokens, start_time, end_time, wait_time))
                # 更新进度条
                pbar.update(1)
                completed_futures.append(future)
                print(f"【当前任务】生成tokens性能: {tokens_per_second:.2f}/s， tokens数量: {total_tokens} 耗时 {latency / 1000:.2f} 秒")
    # 返回包含所有请求结果的列表
    return results

# 将结果写入 CSV 文件（每条结果一列）
def write_to_csv(results, output_file, config):
    """
    将压力测试的结果写入到指定的 CSV 文件中。
    :param results: 包含所有请求结果的列表，每个结果是一个元组，包含响应信息、延迟等。
    :param output_file: 要写入的 CSV 文件的路径。
    :param config: 配置文件内容，虽然当前函数未使用该参数，但可预留扩展。
    """
    # 以写入模式打开指定的 CSV 文件，同时指定编码为 UTF-8，避免中文乱码
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        # 创建一个 CSV 写入器对象，用于后续写入操作
        writer = csv.writer(csvfile)
        # 定义 CSV 文件的表头，用于描述每列数据的含义
        headers = ["测试次数", "响应", "输入 token 数", "输出 token 数", "总 token 数", "耗时（秒）", "首Token延迟(ms)", "非首Token平均延迟(ms)", "Tokens/s", "开始时间", "结束时间", "等待时长（秒）"]
        # 将表头写入到 CSV 文件中
        writer.writerow(headers)
        # 遍历所有的请求结果，同时获取每个结果的索引
        sum_total_tokens = 0
        sum_tokens_per_second = 0
        for i, (response, latency, first_token_latency, non_first_token_avg_latency, tokens_per_second, prompt_tokens, total_tokens, completion_tokens, start_time, end_time, wait_time) in enumerate(results):
            # 格式化开始时间，精确到毫秒
            start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)) + f".{int((start_time - int(start_time)) * 1000):03d}"
            # 格式化结束时间，精确到毫秒
            end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)) + f".{int((end_time - int(end_time)) * 1000):03d}"
            # 将当前结果的各项数据写入到 CSV 文件的一行中
            writer.writerow([
                # 测试次数，从 1 开始计数
                i + 1,
                # 响应内容，从响应结果中提取
                response["choices"][0]["message"]["content"],
                # 输入的 token 数量
                prompt_tokens,
                # 输出的 token 数量
                completion_tokens,
                # 总的 token 数量
                total_tokens,
                # 将响应耗时从毫秒转换为秒，并保留两位小数
                f"{latency / 1000:.2f}",
                # 首 token 延迟，保留两位小数
                f"{first_token_latency:.2f}",
                # 非首 token 平均延迟，保留两位小数
                f"{non_first_token_avg_latency:.2f}",
                # 每秒生成的 token 数量，保留两位小数
                f"{tokens_per_second:.2f}",
                # 格式化后的开始时间
                start_time_str,
                # 格式化后的结束时间
                end_time_str,
                # 等待时长，保留两位小数
                f"{wait_time:.2f}"
            ])
            sum_total_tokens = sum_total_tokens + total_tokens
            sum_tokens_per_second = sum_tokens_per_second + tokens_per_second
        average_tokens = sum_tokens_per_second / config["num_requests"]
        print(f"总生成token数 {sum_total_tokens} 个")
        print(f"平均生成token速度 {average_tokens:.2f} 个/s")

# 主函数，程序的入口点
if __name__ == "__main__":
    # 配置文件的路径
    config_path = "config.json"
    
    try:
        # 记录任务开始时间
        task_start_time = time.time()
        # 打印任务开始时间
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"任务开始时间: {start_time_str}")
        
        # 加载配置文件
        config = load_config(config_path)
        
        # 从配置中获取提示词配置
        prompts = []
        if config.get("prompt_config", {}).get("use_prompt_file", False):
            prompts_file_path = config["prompt_config"]["prompt_file_path"]
            prompts = load_prompts(prompts_file_path)
            print(f"成功加载 {len(prompts)} 个提示词")
        
        # 初始化请求头
        headers = initialize_headers(config)
        # 生成当前时间字符串，用于构建输出文件的名称
        time_str = time.strftime("%y%m%d%H%M%S", time.localtime())
        # 构建输出文件的名称，包含模型名称、请求次数和时间字符串
        output_file = f"压力测试_{config['num_requests']}times_{time_str}.csv"
        # 发起并发请求，并获取所有请求的结果
        results = concurrent_requests(headers, config, prompts)
        # 将结果写入 CSV 文件
        write_to_csv(results, output_file, config)
        
        # 记录任务结束时间
        task_end_time = time.time()
        # 打印任务结束时间
        end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"任务结束时间: {end_time_str}")
        
        # 计算并打印任务总耗时
        task_duration = task_end_time - task_start_time
        print(f"任务总耗时: {task_duration:.2f} 秒")

        # 输出测试完成的信息
        print(f"压力测试完成，结果已写入到 {output_file}")
        
        
    except Exception as e:
        # 输出错误信息
        print(f"运行过程中出现错误: {e}")
        # 导入 traceback 模块，用于获取详细的错误信息
        import traceback
        # 输出详细的错误信息
        print(traceback.format_exc())