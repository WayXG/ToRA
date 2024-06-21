import os
import io
import re
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import multiprocess
from multiprocess import Pool
from typing import Any, Dict, Optional
from pebble import ProcessPool
from tqdm import tqdm
from concurrent.futures import TimeoutError
from functools import partial
from timeout_decorator import timeout
from contextlib import redirect_stdout
import json



NOT_EXECUTED = "<not_executed>"
EXECUTION_ERROR = "Execution error:"
SYNTAX_ERROR = "Syntax error:"
RESULT_NOT_DEFINED_ERROR = "Result is not defined"
TIMEOUT_ERROR = "timeout"
UNDEFINED_ERROR = "Undefined error:"
ERROR_PREFIXES = (EXECUTION_ERROR, SYNTAX_ERROR, RESULT_NOT_DEFINED_ERROR, TIMEOUT_ERROR, UNDEFINED_ERROR)



class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []
    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        # if the code contains input() or os.system(), return Error
        if regex.search(r'(\s|^)?input\(', code_piece) or regex.search(r'(\s|^)?os.system\(', code_piece):
            raise RuntimeError()
        # exec is a built-in python function to execute python code
        # _global_vars is a dict containing the global variables that can be used and modified by the code_piece
        exec(code_piece, self._global_vars)
        
    def eval_code(self, expr: str) -> Any:
        """
        # Evaluate a simple expression
        result = evaluator.eval_code("3 + 4")
        print(result)  # Output: 7

        # Define a variable in the global context and use it in an expression
        evaluator._global_vars['x'] = 10
        result = evaluator.eval_code("x * 2")
        print(result)  # Output: 20

        # Modify a variable in the global context through evaluation
        evaluator.eval_code("x = x + 5")
        print(evaluator._global_vars['x'])  # Output: 15
        """
        return eval(expr, self._global_vars)
    
    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
    
    @property
    def answer(self):
        return self._global_vars['answer']

class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        'datetime': datetime.datetime, 
        'timedelta': dateutil.relativedelta.relativedelta,
        'relativedelta': dateutil.relativedelta.relativedelta
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()

class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {'dict': CustomDict}


class PythonExecutor:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        timeout_length: int = 20,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.pool = Pool(multiprocess.cpu_count())
        self.timeout_length = timeout_length

    def process_generation_to_code(self, gens: str):
        return [g.split('\n') for g in gens]

    @staticmethod
    def execute(
        code,
        get_answer_from_stdout = None,
        runtime = None,
        answer_symbol = None,
        answer_expr = None,
        timeout_length = 10,
    ):
        try:
            if get_answer_from_stdout:
                # io to the memory
                program_io = io.StringIO()
                # redirect_stdout: move all the standard output to the program_io
                with redirect_stdout(program_io):
                    # run the code for at most timeout_length seconds and get all the output to program_io
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                # move the the begging of the outputs
                program_io.seek(0)
                result = program_io.read()
            elif answer_symbol:
                timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                result = runtime._global_vars[answer_symbol]
            elif answer_expr:
                timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                # eval_code(answer_expr), possibly because the global random variables are modified and can be used..
                result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
            else:
                timeout(timeout_length)(runtime.exec_code)('\n'.join(code[:-1]))
                result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            report = "Done"
            str(result)
            pickle.dumps(result) # serialization check
        except:
            #result = ''
            report = traceback.format_exc().split('\n')[-2]
            #result = {{"result": "", "error_message": report}} 
            result = json.dumps({"result":"", "error_message": report})
        return result, report

    def apply(self, code):
        return self.single_batch_apply([code])[0]

    def batch_apply(self, batch_code_seq):
        #batch_results = []
        #print(batch_code_seq)
        all_processed_codes = []
        for code_seq in batch_code_seq:
            
            z = """
import traceback
import json
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '16'

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io
code_snippets = []
"""
            for code_snippet in code_seq:
                z += f'\ncode_snippets.append("""{code_snippet}""")\n'

            z += f"""
try:
    shell = InteractiveShell()
    for tmp_code in code_snippets:
        with io.capture_output() as captured:
            exec_result = shell.run_cell(tmp_code)
    output = f"{{captured.stdout}}{{captured.stderr}}".strip().replace("Out[1]: ", "")    
    error_message = ''
    if exec_result.error_in_exec is not None:
        error_message = f"{EXECUTION_ERROR} {{str(exec_result.error_in_exec)}}"
    elif exec_result.error_before_exec is not None:
        # full traceback will be part of output
        error_message = f"{SYNTAX_ERROR} {{str(exec_result.error_before_exec)}}"
    elif output == "":
        error_message = "{RESULT_NOT_DEFINED_ERROR}"
    to_return = {{"result": output, "error_message": error_message}}
except Exception:
    # removing useless prefix from traceback
    to_return = {{
        "result": None,
        "error_message": "{UNDEFINED_ERROR}" + "\\n".join(traceback.format_exc().split("\\n")[3:]),
    }}
print(json.dumps(to_return))
"""
            #print(z)
            all_processed_codes.append(z)
            #prediction = self.apply(z)
            #print(prediction)
            #dict_data = json.loads(prediction[0])

            #if dict_data['error_message']:
            #    batch_results.append(('', dict_data['error_message']))
            #else:
            #    batch_results.append((dict_data['result'], ''))

        my_results = self.old_batch_apply(all_processed_codes)
        batch_results = []
        for prediction in my_results:
            if prediction[0]:
                try:
                    dict_data = json.loads(prediction[0])
                except:
                    #batch_results.append(('', prediction[1]))
                    #print("json error!!!", prediction)
                    match = re.search(r'"error_message":\s*"([^"]*)"', prediction[0])
                    if match:
                        batch_results.append(('', match.group(1)))
                    else:
                        batch_results.append(('', 'There exists some error in your code. Please rewrite the code and solve the problem.'))
                    print(batch_results[-1])
                    continue
            else:
                batch_results.append(('', prediction[1]))
                continue

            #dict_data = json.loads(prediction[0])
            try:
                dict_data['error_message']
            except:
                batch_results.append(('', 'There exists some error in your code. Please rewrite the code and solve the problem.'))
                print(dict_data)
                continue
            if dict_data['error_message']:
                batch_results.append(('', dict_data['error_message']))
            else:
                batch_results.append((dict_data['result'], ''))
        return batch_results

        
    @staticmethod
    def truncate(s, max_length=800):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s
            
    def old_batch_apply(self, batch_code):

        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []
        with ProcessPool(max_workers=min(len(all_code_snippets), os.cpu_count())) as pool:
            executor = partial(
                self.execute,
                get_answer_from_stdout=self.get_answer_from_stdout,
                runtime=self.runtime,
                answer_symbol=self.answer_symbol,
                answer_expr=self.answer_expr,
                timeout_length=self.timeout_length, # this timeout not work
            )
            future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
            iterator = future.result()

            if len(all_code_snippets) > 100:  
                progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")  
            else:  
                progress_bar = None 

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    #print(error)
                    all_exec_results.append(("", "Timeout Error"))
                    timeout_cnt += 1
                except Exception as error:
                    #print(error)
                    exit()
                if progress_bar is not None:
                    progress_bar.update(1) 
            
            if progress_bar is not None:
                progress_bar.close() 

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # post processing
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            ####
            batch_results.append((res.strip().replace("Out[1]: ", ""), report))
        return batch_results

