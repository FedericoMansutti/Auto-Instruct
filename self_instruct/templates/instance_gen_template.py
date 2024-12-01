output_first_template_for_clf = '''Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate possible class labels.

Task: Classify the sentiment of the sentence into positive, negative, or mixed.
Class label: mixed
Input: I enjoy the flavor of the restaurant but their service is too slow.

Task: Given a dialogue, classify whether the user is satisfied with the service. You should respond with "Satisfied" or "Unsatisfied".
Class label: Satisfied
Input:
- Agent: Thank you for your feedback. We will work to improve our service in the future.
- Customer: I am happy with the service you provided. Thank you for your help.

Task: Given some political opinions, classify whether the person belongs to Democrats or Republicans.
Class label: Democrats
Input: I believe that everyone should have access to quality healthcare regardless of their income level.

Task: Tell me if the following email is a promotion email or not.
Class label: Promotion
Input: Check out our amazing new sale! We've got discounts on all of your favorite products.

Task: Detect if the Reddit thread contains hate speech.
Class label: Hate Speech
Input: All people of color are stupid and should not be allowed to vote.

Task: Answer the following multiple-choice question. Select A, B, C, or D for the final answer.
Class label: C
Input: What is the capital of Germany?
A. London
B. Paris
C. Berlin
D. Rome

Task: Answer the following multiple-choice question. Select A, B, C, or D for the final answer.
Class label: D
Input: What is the largest planet in our solar system?
A) Earth
B) Saturn
C) Mars
D) Jupiter


Task: You need to read a code and detect if there is a syntax error or not. Output true if there is an error, output false if there is not.
Class label: true
Input:
def quick_sort(arr):
    if len(arr) < 2
        return arr

Task: You need to read a code and detect if there is a syntax error or not. Output true if there is an error, output false if there is not.
Class label: False
Input:
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)

Task: You are provided with a news article, and you need to identify all the categories that this article belongs to. Possible categories include Sports and Politics. Output its categories one by one, separated by a comma.
Class label: Sports
Input: The Golden State Warriors have won the NBA championship for the second year in a row.

Task: Given a credit card statement, the cardholder's spending habits, and the account balance, classify whether the cardholder is at risk of defaulting on their payments or not.
Class label: At risk
Input: Purchases at high-end clothing stores and luxury hotels.

Task: Given a social media post, the hashtags used, and a topic. classify whether the post is relevant to the topic or not.
Class label: Relevant
Input: I can't believe the government is still not taking action on climate change. It's time for us to take matters into our own hands.

Task: The answer will be 'yes' if the provided sentence contains an explicit mention that answers the given question. Otherwise, answer 'no'. 
Class label: Yes
Input: Jack played basketball for an hour after school.

Task: Classifying different types of mathematical equations, such as linear, and quadratic equations, based on the coefficients and terms in the equation.
Class label: Linear equation
Input: y = 2x + 5

Task: Tell me the first number of the given list.
Class label: 1
Input: 1, 2, 3

Now it's your turn, remember to output the instance in the same format with no preamble. Follow the same format as the examples above. Let's think step by step.
Task: {instruction}
Class label: {class_labels}
Input:'''

method_first_template_for_gen = '''Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.

Task: Extract all the country names in the paragraph, list them separated by commas.
Input: Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, it was first published in the United Kingdom by Jonathan Cape in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favourably in the United States.
Strategy: Extract country names.
Output: English, British, Jamaica, the United Kingdom, German, Chinese, Britain, the United States.

Task: Converting 85 F to Celsius.
Input: None
Strategy: None.
Output: 85°F = 29.44°C

Task: Sort the given list ascendingly. 
Input: [10, 92, 2, 5, -4, 92, 5, 101]
Strategy: Examine the size of each number.
Output: [-4, 2, 5, 5, 10, 92, 92, 101]

Task: Suggest a better and more professional rephrasing of the following sentence.
Input: This house is surprisingly not constructed very well, and you probably need more money to fix it after you buy it. If you ask me, I would suggest you to consider other candidates.
Strategy: Rephrase the sentence adding professional tone.
Output: This house does not seem to be constructed well, so you may need to spend more money to fix it after you purchase it. I would suggest that you look at other properties.

Task: Read the following paragraph and answer a math question about the paragraph. You need to write out the calculation for getting the final answer.
Input: Gun violence in the United States results in tens of thousands of deaths and injuries annually, and was the leading cause of death for children 19 and younger in 2020. In 2018, the most recent year for which data are available as of 2021, the Centers for Disease Control and Prevention's (CDC) National Center for Health Statistics reports 38,390 deaths by firearm, of which 24,432 were by suicide. The rate of firearm deaths per 100,000 people rose from 10.3 per 100,000 in 1999 to 12 per 100,000 in 2017, with 109 people dying per day or about 14,542 homicides in total, being 11.9 per 100,000 in 2018. In 2010, there were 19,392 firearm-related suicides, and 11,078 firearm-related homicides in the U.S. In 2010, 358 murders were reported involving a rifle while 6,009 were reported involving a handgun; another 1,939 were reported with an unspecified type of firearm. In 2011, a total of 478,400 fatal and nonfatal violent crimes were committed with a firearm. How many more firearm-related deaths were there in 2018 compared to 2010?
Strategy: Calculate the number of firearm-related deaths in 2018 compared to 2010.
Output:
38390 - (19392 + 11078) = 38390 - 30470 = 7920. 
So, in 2018, there were 7920 more deaths by firearm than in 2010.

Task: Write Python code to solve this leetcode problem.
Input: You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.
Strategy: Add the two numbers and return the sum as a linked list.
Output:
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next

Task: Select the oldest person from the given list.
Input: George Washington, Confucius, Michael Jordan, Michelangelo
Strategy: Compare the birth years of the people.
Output: Confucius

Now it's your turn, remember to output in the same format with no preamble. Follow the same format as the examples above.
Keep the answers extremely CONCISE and to the point. Let's think step by step.
Instruction: {instruction}
Input: {input}
Strategy: {strategy}
Output:'''