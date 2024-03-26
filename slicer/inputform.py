from django import forms


class inputForm(forms.Form):
    size = forms.IntegerField()
    absorption = forms.FloatField()
    y = forms.IntegerField()
    gap_num = forms.IntegerField()
    gap_size = forms.IntegerField()
    gap_space = forms.IntegerField()
    potential_type = forms.CharField(max_length=50)