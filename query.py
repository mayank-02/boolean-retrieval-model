from Stack import Stack


def precedence(token):
    """ Precedence of supported operators """
    __precedence = {"&": 2, "|": 1}
    try:
        return __precedence[token]
    except:
        return -1


def is_left_bracket(token):
    """ Returns true if left bracket """
    return token == "("


def is_right_bracket(token):
    """ Returns true if right bracket """
    return token == ")"


def is_operator(token):
    """ Returns true if operator """
    return token == "&" or token == "|"


def infix_to_postfix(tokens):
    """Converts a infix query into postfix
    Input : ['god', '&', '(', '~child', '|', 'mother', ')']
    Output : ['god', '~child', 'mother', '|', '&']

    :param tokens: list of tokens in infix form
    :returns: same list of tokens in postfix form
    """

    stack = Stack()
    postfix = list()

    for token in tokens:

        if is_left_bracket(token):
            # Left bracket "("
            stack.push(token)

        elif is_right_bracket(token):
            # Right bracket ")"
            while (not stack.is_empty()) and stack.peek() != "(":
                key = stack.pop()
                postfix.append(key)
            if not stack.is_empty() and stack.peek() != "(":
                raise ValueError("Query isn't well formatted")
            else:
                stack.pop()

        elif is_operator(token):
            # Operator
            while not stack.is_empty() and (
                precedence(token) <= precedence(stack.peek())
            ):
                postfix.append(stack.pop())
            stack.push(token)

        else:
            # Operand
            postfix.append(token)

    # Pop all the operator from the stack
    while not stack.is_empty():
        postfix.append(stack.pop())

    return postfix