#!/usr/bin/env python3

from typing import Union, Callable, List

class ExpTreeNode:
    """ An abstract class, representing the tree node
    of an expression tree. 

    """

    def evaluate(self) -> Union[int, float]: 
        """ Evaluate the expression tree starting
        from this node. Returns a numerical value
        """
        raise NotImplementedError

    @classmethod
    def parse(cls, exp: str) -> "ExpTreeNode": 
        """ Parse the given string into an expression tree

        Params
        ------
        exp: the input expression

        Returns
        -------
        The root node

        """
        raise NotImplementedError

    @classmethod
    def _split_exp_into_nodes(cls, exp: str) -> List[Union[str, List]]:
        """ Split the expression into list of elements

        Params
        ------
        exp: the input expression

        Returns
        -------
        A list of elements, each of which is either
        * a string for an operator or an operand, or
        * a list as a sub-expression
        """
        ch_list = []
        skip = 0
        SHRINKING_MAP = {
            '\\left{' : '{',
            '\\left(' : '(',
            '\\left[' : '[',
            '\\right}' : '}',
            '\\right)' : ')',
            '\\right]' : ']', 
            '\\frac' : 'F'
        }
        """The escape combination"""
        LEFT_RIGHT_MATCHING = {
            '{' : '}',
            '[' : ']',
            '(' : ')'
        }
        """The parentheses"""

        # STEP ONE: shrink the escape combinations
        for idx, ch in enumerate(exp):
            if skip > 0:
                skip -= 1
                continue
            if ch == '\\':
                while idx + skip < len(exp) and exp[idx + skip] != ' ' and exp[idx + skip] != '{' and not exp[idx + skip].isdigit():
                    skip += 1
                if idx + skip <= len(exp) and exp[idx : idx + skip] in SHRINKING_MAP:
                    ch_list.append(SHRINKING_MAP[exp[idx : idx + skip]])
                else:
                    raise ValueError(f"invalid escape expression {exp[idx : idx + skip]}")
                skip -= 1
            else: 
                ch_list.append(ch)

        # print(exp, '->', ','.join(ch_list))

        # STEP TWO: recusively form the groups

        group_list = []
        skip = 0
        
        for idx, ch in enumerate(ch_list):
            if skip != 0: 
                skip -= 1
                continue

            if ch in LEFT_RIGHT_MATCHING: 
                stack = [ch]
                skip += 1
                while idx + skip < len(ch_list): 
                    # print(ch_list[idx + skip], '<->', stack)
                    if ch_list[idx + skip] in LEFT_RIGHT_MATCHING:
                        stack.append(ch_list[idx + skip])
                    elif stack and LEFT_RIGHT_MATCHING[stack[-1]] == ch_list[idx + skip]: 
                        stack.pop()

                    skip += 1

                    if not stack: 
                        group_list.append(cls._split_exp_into_nodes(ch_list[idx + 1 : idx + skip - 1]))
                        skip -= 1
                        break

                else:
                    if stack: 
                        raise ValueError(f'Unmatched {{ and }}, there are {stack}')

            elif ch.isdigit(): 
                if group_list and type(group_list[-1]) == str and group_list[-1].isdigit():
                    origin = group_list.pop()
                    group_list.append(origin + ch)
                else:
                    group_list.append(ch)
            
            elif ch == ' ':
                continue
            
            else:
                group_list.append(ch)

        return group_list


class NumericalNode(ExpTreeNode):
    """ The expression tree node for a numerical value,
    either a float number or an integer

    Params
    ------
    val: the numerical value the node is representing
    """
    def __init__(self, val: Union[int, float]):
        self._val = val

    def evaluate(self) -> Union[int, float]: 
        return self._val

    @classmethod
    def parse(cls, exp:str) -> "NumericalNode":
        try:
            val = int(exp)
        except ValueError:
            try: 
                val = float(exp)
            except ValueError:
                raise ValueError(f'{exp} is not an int or a float')
        return NumericalNode(val)

class OperatorNode(ExpTreeNode):
    """ An expression tree node for a operator

    Params
    -----
    opt: a callable, whose input shall be the operator's
            operands, returning the results. The nested class
            `PreDefinedOperations` offers predefined algebra
            operations. It is suggested to use the predefined
            operations ONLY, as the priorities are clearly
            specified for those predefined operations. 
    """

    class PreDefinedOperations:
        """ The predefined operations, for the convenience of
        initialization the OperatorNode
        """
        @classmethod
        def divide(cls, children: List[ExpTreeNode]):
            assert len(children) == 2, f"There must be 2 operands for division, got {len(children)}"
            return children[0].evaluate() / children[1].evaluate()

        @classmethod
        def multiply(cls, children: List[ExpTreeNode]):
            assert len(children) == 2, f"There must be 2 operands for multiplication, got {len(children)}"
            return children[0].evaluate() * children[1].evaluate()

        @classmethod
        def add(cls, children: List[ExpTreeNode]):
            assert len(children) == 2, f"There must be 2 operands for addition, got {len(children)}"
            return children[0].evaluate() + children[1].evaluate()

        @classmethod
        def substract(cls, children: List[ExpTreeNode]):
            assert len(children) == 2, f"There must be 2 operands for substraction, got {len(children)}"
            return children[0].evaluate() - children[1].evaluate()

        @classmethod
        def power(cls, children: List[ExpTreeNode]):
            assert len(children) == 2, f"There must be 2 operands for power, got {len(children)}"
            return children[0].evaluate() ** children[1].evaluate()

        @classmethod
        def get_operation_func(cls, operator: str) -> Callable[[List[ExpTreeNode]], Union[int, float]]:
            """ Find the predefined func corresponding to the given operator string

            Params
            ------
            operator: the operator string

            Returns
            -------
            A callable, instructing how the operator should evaluate its operands
            -----
            """
            if operator == '+': return cls.add
            if operator == '-': return cls.substract
            if operator == '*': return cls.multiply
            if operator == '/': return cls.divide
            if operator == '^': return cls.power

            raise ValueError(f"unsupported operator: {operator}")

        @classmethod
        def op_func_priority(cls, operator: Callable) -> int:
            """Defining the priority of the predefined operator functions
            
            The higher the returned value is, the higher priority it has
            """
            if operator == cls.power: return 5
            if operator == cls.divide or operator == cls.multiply: return 3
            if operator == cls.add or operator == cls.substract: return 1


    def __init__(self, opt: Callable[[List[ExpTreeNode]], Union[int, float]]):
        self._inner_opt = opt
        self.children = []

    def evaluate(self) -> Union[int, float]: 
        val = self._inner_opt(self.children)
        if isinstance(val, int) or val == int(val): return int(val)
        return val

    def __gt__(self, obj: "OperatorNode"):
        assert isinstance(obj, OperatorNode)
        return self.PreDefinedOperations.op_func_priority(self._inner_opt) > \
                self.PreDefinedOperations.op_func_priority(obj._inner_opt)
        
    
    @classmethod
    def _parse(cls, exp: List[Union[str, List]]) -> ExpTreeNode:
        """ The inner implementation for the cls.parse. 

        This implementation, instead of taking a string, requires pre-
        processed list of elements (the returned value from
        `ExpTreeNode._split_exp_into_nodes`), and returns the root
        node
        """
        skip = 0
        stack = []
        hold_operand = None
        for idx, elem in enumerate(exp):
            if skip > 0:
                skip -= 1
                continue

            if isinstance(elem, list): 
                node = cls._parse(elem)
            elif elem == 'F':
                node = OperatorNode(cls.PreDefinedOperations.divide)
                # print('init F operator')
                assert len(exp) > idx + 2, "no enough operands for division"
                node.children.append(cls._parse(exp[idx + 1]))
                # print(f'node.children is {node.children[0].evaluate()}')
                node.children.append(cls._parse(exp[idx + 2]))
                # print(f'node.children is {node.children[1].evaluate()}')
                skip += 2
            elif elem.isdigit():
                node = NumericalNode.parse(elem)
            else:
                # NONE <=> Operator!
                node = None

            if node != None:
                hold_operand = node
            else:
                # this is an operator! 
                node = OperatorNode(cls.PreDefinedOperations.get_operation_func(elem))
                if not stack:
                    stack.append(node)
                    node.children.append(hold_operand)
                else:
                    while stack[-1] > node: 
                        prev_node = stack.pop()
                        prev_node.children.append(hold_operand)
                        hold_operand = prev_node
                    node.children.append(hold_operand)
                    stack.append(node)
        while stack: 
            node = stack.pop()
            node.children.append(hold_operand)
            hold_operand = node
        return hold_operand


    @classmethod
    def parse(cls, exp:str) -> ExpTreeNode:
        exp_list = ExpTreeNode._split_exp_into_nodes(exp)
        return cls._parse(exp_list)



def evaluate_exp(exp: str) -> Union[int, float, None]:
    """ Evaluate the expression

    Return the numerical result, or None if the expression
    is invalid or not supported

    """
    return OperatorNode.parse(exp).evaluate()


