import ast
p='src/emergent/salmon_abm/behavior.py'
s=open(p,'r',encoding='utf-8').read()
try:
    ast.parse(s)
    print('AST OK')
except SyntaxError as e:
    print('SyntaxError', e.msg, 'line', e.lineno, 'offset', e.offset)
