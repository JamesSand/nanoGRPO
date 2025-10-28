from math_verify import parse, verify

# Parse the gold and answer
# If you know that gold will only contain latex or expr (no latex env), use
# parse(gold, extraction_config=[LatexExtractionConfig()]) or parse(gold, extraction_config=[ExprExtractionConfig()])

gold = parse("The answer is \\boxed{${1,3} \\cup {2,4}$} What if I add some bad in the end")
answer = parse("${1,2,3,4}$")

print(gold)
print(answer)

# Order here is important!
print(verify(gold, answer))
# >>> True



