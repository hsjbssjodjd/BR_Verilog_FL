from openai import OpenAI

# API 设定
API_KEY = "sk-a08b162df7824823865daef1b8cd7ce4"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


verilog_conversion_prompt = """**角色**：您是一位精通Verilog到Yosys转换的芯片验证专家，擅长将现代Verilog/SystemVerilog代码降级为Yosys兼容的的Verilog-2005代码。
我将为您提供完整的bug报告和需要转换得Verilog代码,请您根据bug报告的信息将提供的代码转换成新的Yosys兼容的Verilog-2005代码，以下转换规则提供参考：

━━━ 转换规则（按优先级排序） ━━━
1. 【语法降级】
   ✓ 将`logic`类型替换为`reg`/`wire`
   ✓ 转换`always_comb` → `always @(*)`
   ✓ 转换`always_ff` → `always @(posedge clk)`
   ✓ 转换`always_latch` →`always @(*)` + `if (enable)` 逻辑 
   ✗ 移除`interface`、`package``typedef`、`import`等不可综合结构
  

2. 【行为转换】
   ✓ 保持组合逻辑功能等效性
   ✓ 将锁存器转换为`reg`+使能逻辑（示例：latch → reg + enable）
   ✓ 转换`unique case` / `priority case` → 为普通 `case` 语句
   ✓ 转换`struct` / `enum` → `parameter` / `localparam` 
   ✗ 移除`initial`块中的时序控制（保留寄存器初始化）
   ✗ 删除所有`#delay`语句

3. 【验证辅助】
   ✓ 将`$display`替换为注释（格式：`// DISPLAY: 原始内容`）
   ✓ 保留信号名称和模块层次结构
   ✓ 转换`task` / `function` →为 Verilog-2005 兼容函数 
   ✓ 所有 `input/output` 端口应显式声明 `wire` 或 `reg` 类型


**需要转换的代码**：
{verilog_code}

**输出格式要求**：
- 仅返回 转换后的 Verilog-2005 代码
- 禁止任何解释/附加内容
- 禁止使用 SystemVerilog 语法

"""


# 调用 API 生成测试用例
def generate_unit_test(prompt):
    try:
        completion = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        

        print("调用大模型成功")
        # 通过reasoning_content字段打印思考过程
#        print("思考过程：")
#        print(completion.choices[0].message.reasoning_content)
#
#        # 通过content字段打印最终答案
#        print("最终答案：")
#        print(completion.choices[0].message.content)
        return completion.choices[0].message.content  # 返回完整的生成内容
    except Exception as e:
        print(f"调用大模型失败: {e}")
        return ""


# 主执行流程
def verilog_conversion(verilog_code):

    prompt = verilog_conversion_prompt.replace("{verilog_code}", verilog_code)
    # 去除 prompt 里的空行
    cleaned_prompt = "\n".join([line for line in prompt.split("\n") if line.strip()])
    print("**********",cleaned_prompt)

    # 调用 大模型 生成测试用例
    response_text = generate_unit_test(cleaned_prompt)

    # 直接保存大模型返回的完整内容
    if response_text:
        print("+++++++++++")
        return response_text
#        with open("gptVerilog_file.v", "w", encoding="utf-8") as file:
#            file.write(response_text)
#        print("转换代码已经生成: gptVerilog_file.v")
    else:
        print("未能生成有效的转换代码")
